"""Analyst dashboard (Streamlit) — HITL verification console, intel UX.

Run: `streamlit run dashboard/app.py` (needs `pip install streamlit requests`).
Talks only to the FastAPI (`API_BASE`); the air-gap boundary is unchanged.
Every panel degrades with an explicit message when its backend leg is
unavailable — never a silent blank panel.

Layout: TLP session banner → sidebar (health, clearance, metrics) →
wizard tabs Ingest → Review (per-leg sub-tabs) → HITL → OSINT.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import time

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8088")

try:
    import requests
    import streamlit as st
except ImportError:
    print("dashboard needs: pip install streamlit requests")
    raise SystemExit(1) from None


# ---------------------------------------------------------------- theme helpers

_TLP_COLORS = {
    "TLP:CLEAR": "#9aa4b2",
    "TLP:GREEN": "#2ea043",
    "TLP:AMBER": "#d29922",
    "TLP:RED": "#f85149",
}
_VERDICT_COLORS = {
    "AUTHENTIC": "#2ea043",
    "LOW_RISK": "#7ee787",
    "UNVERIFIABLE": "#58a6ff",
    "SUSPICIOUS": "#d29922",
    "HIGH_RISK": "#f85149",
}
_STATUS_STEPS = ["queued", "running", "done"]


def _css() -> None:
    st.markdown(
        """<style>
        .tlp-banner { padding: 6px 14px; border-radius: 6px; font-weight: 700;
                      letter-spacing: .06em; margin-bottom: 12px; color: #0d1117; }
        .verdict-banner { padding: 10px 16px; border-radius: 8px; font-size: 1.15em;
                          font-weight: 700; margin: 8px 0 4px 0; color: #0d1117; }
        .gauge { width: 148px; height: 148px; border-radius: 50%; display: flex;
                 align-items: center; justify-content: center; margin: 4px auto; }
        .gauge-inner { width: 112px; height: 112px; border-radius: 50%;
                       background: #0d1117; display: flex; flex-direction: column;
                       align-items: center; justify-content: center; color: #f0f6fc; }
        .kv { font-size: .92em; color: #8b949e; }
        .step-done { color: #2ea043; font-weight: 700; }
        .step-todo { color: #484f58; }
        </style>""",
        unsafe_allow_html=True,
    )


def _tlp_banner(tlp: str, text: str = "") -> None:
    color = _TLP_COLORS.get(tlp, "#9aa4b2")
    st.markdown(
        f'<div class="tlp-banner" style="background:{color}">'
        f"{tlp}{' — ' + text if text else ''}</div>",
        unsafe_allow_html=True,
    )


def _verdict_banner(verdict: str, risk: int) -> None:
    color = _VERDICT_COLORS.get(verdict, "#9aa4b2")
    st.markdown(
        f'<div class="verdict-banner" style="background:{color}">{verdict} · risk {risk}/100</div>',
        unsafe_allow_html=True,
    )


def _risk_gauge(risk: int) -> None:
    color = (
        "#f85149"
        if risk >= 60
        else "#d29922"
        if risk >= 40
        else "#58a6ff"
        if risk >= 20
        else "#2ea043"
    )
    st.markdown(
        f'<div class="gauge" style="background: conic-gradient({color} '
        f'{risk}%, #21262d {risk}%);">'
        f'<div class="gauge-inner"><div style="font-size:1.7em;font-weight:800;">'
        f'{risk}</div><div class="kv">/ 100</div></div></div>',
        unsafe_allow_html=True,
    )


def _stepper(status: str) -> None:
    idx = _STATUS_STEPS.index(status) if status in _STATUS_STEPS else 0
    cols = st.columns(3)
    for i, name in enumerate(["Ingested", "Analyzing", "Done"]):
        cls = (
            "step-done"
            if (i <= idx and status == "done") or i < idx
            else ("step-done" if i == idx else "step-todo")
        )
        mark = "●" if (i < idx or status == "done") else ("◐" if i == idx else "○")
        cols[i].markdown(f'<span class="{cls}">{mark} {name}</span>', unsafe_allow_html=True)
    if status == "failed":
        st.error("Analysis failed — see worker logs; case retained for triage.")


st.set_page_config(page_title="Provenance Verification Console", layout="wide")
_css()
st.title("Media Provenance & Verification — Analyst Console")


# ---------------------------------------------------------------- sidebar: session + health


def _api_health(base: str) -> tuple[bool, str]:
    try:
        r = requests.get(base + "/health", timeout=5)
        if r.status_code == 200:
            body = r.json()
            dev = (body.get("compute") or {}).get("torch_device", "?")
            return True, f"API ok · torch:{dev}"
        return False, f"API http {r.status_code}"
    except Exception as exc:
        return False, f"API down: {exc}"[:90]


with st.sidebar:
    st.header("Session")
    clearance = st.selectbox("Your clearance", ["TLP:CLEAR", "TLP:GREEN", "TLP:AMBER", "TLP:RED"])
    analyst = st.text_input("Analyst callsign", "analyst")
    api_base = st.text_input("API base", API_BASE)
    ok, msg = _api_health(api_base)
    st.markdown("🟢 " + msg if ok else "🔴 " + msg)
    with st.expander("Service metrics"):
        try:
            m = requests.get(api_base + "/metrics", timeout=8).json().get("metrics", {})
            st.write(dict(m))
        except Exception:
            st.caption("metrics unavailable (older API)")
    st.caption("TLP gates every record. All actions are audit-logged.")

_tlp_banner(clearance, "session clearance — records above this are hidden")


def _get(path: str, **kw):
    try:
        return requests.get(api_base + path, headers={"X-TLP": clearance}, timeout=30, **kw)
    except Exception as exc:
        st.error(f"API unreachable: {exc}")
        return None


def _fetch_case(cuuid: str):
    r = _get(f"/api/v1/cases/{cuuid}", params={"clearance": clearance})
    if r is None:
        return None
    if r.status_code == 200:
        return r.json()
    if r.status_code == 403:
        st.warning("Clearance insufficient for this case's TLP.")
    else:
        st.error(f"case fetch failed {r.status_code}: {r.text[:300]}")
    return None


# ---------------------------------------------------------------- 1 · ingest

tab_up, tab_case, tab_hitl, tab_osint = st.tabs(
    ["① Ingest", "② Case review", "③ HITL feedback", "④ OSINT feed"]
)

with tab_up:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Evidence file")
        up = st.file_uploader(
            "Drag-and-drop image / video / audio",
            type=[
                "jpg",
                "jpeg",
                "png",
                "tif",
                "tiff",
                "webp",
                "mp4",
                "mov",
                "mkv",
                "avi",
                "webm",
                "wav",
            ],
        )
        if up is not None:
            raw = up.getvalue()
            sha = hashlib.sha256(raw).hexdigest()
            meta_l, meta_r = st.columns([1, 1])
            with meta_l:
                st.code(f"SHA-256\n{sha}", language="text")
                st.caption(f"{len(raw) / 1024:.1f} KiB · {up.name} · {up.type or 'unknown type'}")
            with meta_r:
                if (up.type or "").startswith("image/"):
                    try:
                        from PIL import Image as _PI

                        img = _PI.open(io.BytesIO(raw)).convert("RGB")
                        img.thumbnail((360, 360))
                        st.image(img, caption=f"{img.size[0]}×{img.size[1]} px preview")
                    except Exception:
                        st.caption("preview unavailable")
                elif (up.type or "").startswith("audio/"):
                    st.audio(raw)
                else:
                    st.caption("preview: images and audio only (video plays after ingest)")
    with right:
        st.subheader("Handling")
        tlp = st.selectbox(
            "Classification (TLP)",
            ["TLP:CLEAR", "TLP:GREEN", "TLP:AMBER", "TLP:RED"],
            index=2,
            help="Unvetted field media defaults to AMBER or above.",
        )
        _tlp_banner(tlp, "selected classification")
        with st.expander("Claimed capture (GEOINT, optional)"):
            lat = st.text_input("Latitude", "")
            lon = st.text_input("Longitude", "")
            ts = st.text_input("Capture UTC (ISO-8601)", "")
        submit = st.button(
            "Submit for verification", type="primary", disabled=up is None, use_container_width=True
        )
    if up is not None and submit:
        raw = up.getvalue()
        files = {"file": (up.name, io.BytesIO(raw), up.type or "application/octet-stream")}
        data = {"tlp": tlp, "owner": analyst}
        if lat and lon and ts:
            data.update({"latitude": lat, "longitude": lon, "timestamp_utc": ts})
        try:
            r = requests.post(
                api_base + "/api/v1/verify",
                files=files,
                data=data,
                headers={"X-TLP": clearance},
                timeout=120,
            )
            if r.status_code == 200:
                body = r.json()
                st.session_state["case_uuid"] = body["case_uuid"]
                st.success(
                    f"Case `{body['case_uuid'][:8]}…` accepted — "
                    f"task `{str(body['task_id'])[:8]}…`. Open ② Case review."
                )
            elif r.status_code == 429:
                st.warning("Rate limited — wait a minute and retry.")
            else:
                st.error(f"ingest failed {r.status_code}: {r.text[:400]}")
        except Exception as exc:
            st.error(f"submit failed: {exc}")

# ---------------------------------------------------------------- 2 · review

with tab_case:
    st.subheader("Case review")
    c1, c2 = st.columns([4, 1])
    with c1:
        cuuid = st.text_input(
            "Case UUID",
            st.session_state.get("case_uuid", ""),
            label_visibility="collapsed",
            placeholder="paste case UUID…",
        )
    with c2:
        if st.button("Open", use_container_width=True) and cuuid:
            st.session_state["case_uuid"] = cuuid
    cuuid = st.session_state.get("case_uuid", "")
    if not cuuid:
        st.info("Submit evidence in ① Ingest, or paste a case UUID.")
    else:
        case = _fetch_case(cuuid)
        if case:
            _tlp_banner(
                case.get("tlp", "TLP:CLEAR"),
                f"case {cuuid[:8]}… · `{case.get('sha256', '')[:16]}…`",
            )
            _stepper(case.get("status", "queued"))
            if case.get("status") not in ("done", "failed"):
                bar = st.progress(0.15, text="Analysis running — polling…")
                for i in range(20):
                    time.sleep(4)
                    case = _fetch_case(cuuid) or case
                    bar.progress(min(0.15 + 0.04 * (i + 1), 0.95))
                    if case.get("status") in ("done", "failed"):
                        break
                bar.empty()
                _stepper(case.get("status", "queued"))
            res = case.get("result") or {}
            if not res:
                st.info("No findings yet — still queued/running, or withheld at your clearance.")
            else:
                risk = int(res.get("risk_score", 0))
                _verdict_banner(str(res.get("verdict", "")), risk)
                k1, k2, k3, k4 = st.columns([1, 1, 1, 2])
                with k1:
                    _risk_gauge(risk)
                with k2:
                    st.metric("Confidence", res.get("analytic_confidence", "?"))
                    st.metric("LLM backend", res.get("llm_backend", "?"))
                with k3:
                    st.metric("Schema", res.get("schema_version", "?"))
                    st.metric("TLP", res.get("tlp", "?"))
                with k4:
                    st.markdown("**Executive summary**")
                    st.write(res.get("executive_summary", ""))
                nt, fo, au, bi, ge, cx, ex = st.tabs(
                    [
                        "Narrative",
                        "Forensics",
                        "Audio & Speech",
                        "Biometrics",
                        "GEOINT",
                        "Context",
                        "Export",
                    ]
                )
                with nt:
                    st.write(res.get("narrative", "_withheld at your clearance_"))
                    hyps = res.get("hypotheses", [])
                    if hyps:
                        st.markdown("**Competing hypotheses (ACH)**")
                        with contextlib.suppress(Exception):
                            st.bar_chart(
                                {
                                    str(h.get("name", "?")).split("—")[-1].strip()[:28]: float(
                                        h.get("likelihood", 0)
                                    )
                                    for h in hyps
                                },
                                horizontal=True,
                            )
                        for h in hyps:
                            st.caption(
                                f"{h.get('name')} — {h.get('term')} (p≈{h.get('likelihood')})"
                            )
                    with st.expander("Findings & recommendations"):
                        for f in res.get("findings", []):
                            st.write(f"- {f}")
                        for x in res.get("recommendations", []):
                            st.write(f"+ {x}")
                with fo:
                    fr = res.get("forensic") or {}
                    st.caption(
                        f"anomaly {fr.get('aggregate_anomaly')} · "
                        f"{fr.get('num_frames_analyzed')} frame(s) · "
                        f"{fr.get('device_used')} · onnx={fr.get('onnx_used')} · "
                        f"temporal={fr.get('temporal_consistency')}"
                    )
                    frames = fr.get("frames", [])
                    if frames:
                        st.dataframe(
                            [
                                {
                                    k: f.get(k)
                                    for k in (
                                        "frame_index",
                                        "anomaly_score",
                                        "ela_score",
                                        "fft_score",
                                        "blockiness",
                                        "dct_double_jpeg",
                                        "face_count",
                                        "face_anomaly",
                                        "orb_clusters",
                                    )
                                }
                                for f in frames
                            ],
                            use_container_width=True,
                        )
                    for w in fr.get("warnings", []):
                        st.warning(w)
                    mp = (res.get("multimodal") or {}).get("mp4") or {}
                    if mp.get("is_mp4"):
                        st.caption(
                            f"MP4 atoms: {len(mp.get('atoms', []))} · "
                            f"faststart={mp.get('faststart')} · "
                            f"double-compression hint "
                            f"{mp.get('double_compression_hint')}"
                        )
                        for a in mp.get("anomalies", []):
                            st.warning(a)
                    wm = (res.get("multimodal") or {}).get("watermark") or {}
                    if wm.get("metadata_hits") or wm.get("ai_generated"):
                        st.error(
                            f"AI-generated: {wm.get('ai_generated')} · "
                            f"generator: {wm.get('generator')} · "
                            f"SynthID: {wm.get('synthid')}"
                        )
                        for h in wm.get("metadata_hits", [])[:5]:
                            st.code(h)
                with au:
                    au = (res.get("multimodal") or {}).get("audio") or {}
                    if au.get("available"):
                        st.metric("Vocoder anomaly", au.get("vocoder_anomaly"))
                        st.metric("Pitch-flatness flag", au.get("pitch_instability"))
                        for n in au.get("spectral_notes", []):
                            st.caption(n)
                        for w in au.get("warnings", []):
                            st.warning(w)
                    else:
                        st.caption("; ".join(au.get("warnings", ["no audio track"])))
                    ls = (res.get("multimodal") or {}).get("lipsync") or {}
                    if ls.get("available"):
                        st.metric("Lip-sync", f"{ls.get('verdict')} (r={ls.get('correlation')})")
                    sp = (res.get("multimodal") or {}).get("speech") or {}
                    if sp.get("available"):
                        st.markdown(
                            f"**Transcript** ({sp.get('language', '?')}, "
                            f"p={sp.get('language_probability')})"
                        )
                        st.write(sp.get("text", ""))
                        if sp.get("english_gloss"):
                            with st.expander("English translation"):
                                st.write(sp["english_gloss"])
                    elif sp.get("errors"):
                        st.caption("; ".join(sp["errors"][:1]))
                with bi:
                    rp = (res.get("multimodal") or {}).get("rppg") or {}
                    if rp.get("available"):
                        st.metric("Heart rate", f"{rp.get('heart_rate_bpm')} bpm")
                        st.metric(
                            "Liveness", f"{rp.get('liveness')} (q={rp.get('signal_quality')})"
                        )
                        if rp.get("waveform"):
                            st.line_chart(rp["waveform"])
                        for w in rp.get("warnings", []):
                            st.warning(w)
                    else:
                        st.caption(
                            "; ".join(
                                rp.get("errors") or rp.get("warnings") or ["rPPG needs face video"]
                            )
                        )
                with ge:
                    geo = (res.get("multimodal") or {}).get("geoint") or {}
                    if geo.get("available"):
                        g1, g2, g3 = st.columns(3)
                        g1.metric(
                            "Sun elev/az",
                            f"{geo.get('solar_elevation_deg')}° / {geo.get('solar_azimuth_deg')}°",
                        )
                        g2.metric("Shadow error", f"{geo.get('angular_error_deg')}°")
                        g3.metric("Consistent", str(geo.get("consistent")))
                        if geo.get("latitude") is not None:
                            st.map(data=[{"lat": geo["latitude"], "lon": geo["longitude"]}])
                        for w in geo.get("warnings", []):
                            st.warning(w)
                    else:
                        st.caption("GEOINT needs claimed lat/lon + UTC at ingest.")
                with cx:
                    ctx = res.get("context", {}) if isinstance(res.get("context"), dict) else {}
                    st.caption(
                        f"backend {(res.get('context') or {}).get('backend', '?')} · "
                        f"phash {(res.get('context') or {}).get('query_phash', '')[:12]}… · "
                        "clip="
                        f"{'yes' if (res.get('context') or {}).get('clip_available') else 'no'}"
                    )
                    r4 = _get(
                        "/api/v1/search/similar",
                        params={"case_uuid": cuuid, "clearance": clearance},
                    )
                    if r4 and r4.status_code == 200:
                        corr = r4.json().get("correlated", [])
                        if corr:
                            st.dataframe(
                                [
                                    {
                                        "event": c.get("event_id"),
                                        "score": c.get("score"),
                                        "tier": c.get("tier", c.get("match_type", "?")),
                                        "note": str(c.get("description", ""))[:90],
                                    }
                                    for c in corr[:15]
                                ],
                                use_container_width=True,
                            )
                        else:
                            st.caption("no correlations — check RAG citations below.")
                    for c in res.get("citations", [])[:5]:
                        st.caption(f"ref: {c}")
                with ex:
                    if st.button("Export STIX 2.1 bundle", type="primary"):
                        r3 = _get(f"/api/v1/export/stix/{cuuid}", params={"clearance": clearance})
                        if r3 and r3.status_code == 200:
                            st.download_button(
                                "Download bundle",
                                json.dumps(r3.json(), indent=2),
                                file_name=f"{cuuid}.stix.json",
                                mime="application/json",
                            )
                            st.caption("MISP-ready indicators + SOC JSON in one bundle.")
                        else:
                            st.error(f"export failed: {r3.status_code if r3 else 'no API'}")
                    rd = (res.get("multimodal") or {}).get("redaction") or {}
                    if rd.get("redacted_path"):
                        st.success(
                            f"PII-redacted copy: `{rd['redacted_path']}` "
                            f"(faces {rd.get('faces_blurred')}, "
                            f"plates {rd.get('plates_blurred')})"
                        )
            with st.expander("Chain-of-custody audit"):
                trail = case.get("audit", [])[-20:]
                if trail:
                    st.dataframe(
                        [
                            {"ts": a.get("ts"), "actor": a.get("actor"), "action": a.get("action")}
                            for a in trail
                        ],
                        use_container_width=True,
                    )
                else:
                    st.caption("no trail entries visible at your clearance.")

# ---------------------------------------------------------------- 3 · HITL

with tab_hitl:
    st.subheader("Analyst correction → Active Learning")
    cuuid = st.session_state.get("case_uuid", "")
    if not cuuid:
        st.info("Submit or open a case first.")
    else:
        case = _fetch_case(cuuid)
        if case and case.get("result"):
            st.caption(
                f"Current verdict: **{case['result'].get('verdict')}** "
                f"(risk {case['result'].get('risk_score')}) — "
                f"confirm it or flag the error class."
            )
        label = st.radio(
            "Correction",
            ["CONFIRM", "FP", "FN"],
            format_func=lambda x: {
                "CONFIRM": "CONFIRM — verdict stands",
                "FP": "FP — false positive (over-called risk)",
                "FN": "FN — false negative (missed manipulation)",
            }[x],
        )
        finding_ref = st.text_input(
            "Finding reference", placeholder="e.g. H2, face anomaly 0.81, EVT-2020-BEIRUT…"
        )
        notes = st.text_area(
            "Analyst notes",
            placeholder="what contradicts the finding? (encrypted at rest when configured)",
        )
        if st.button("Submit to Active Learning queue", type="primary"):
            try:
                r = requests.post(
                    api_base + "/api/v1/feedback",
                    json={
                        "case_uuid": cuuid,
                        "label": label,
                        "analyst": analyst,
                        "finding_ref": finding_ref,
                        "notes": notes,
                    },
                    timeout=30,
                )
                if r.status_code == 200:
                    st.success(
                        f"Feedback `{r.json()['feedback_id']}` queued — "
                        f"retraining dataset stages at threshold."
                    )
                else:
                    st.error(f"feedback failed {r.status_code}: {r.text[:300]}")
            except Exception as exc:
                st.error(f"submit failed: {exc}")

# ---------------------------------------------------------------- 4 · OSINT

with tab_osint:
    st.subheader("OSINT ingestion (Telegram / RSS)")
    st.caption(
        "Unvetted OSINT defaults to TLP:AMBER. Beat task "
        "`provenance.osint_poll` runs on schedule; trigger manually here."
    )
    if st.button("Run ingestion cycle now", type="primary"):
        try:
            with st.spinner("Polling feeds…"):
                r = requests.post(api_base + "/api/v1/osint/poll", timeout=300)
            if r.status_code == 200:
                body = r.json()
                st.success(f"{body['created']} new case(s)")
                for c in body.get("cases", [])[:20]:
                    row_l, row_r = st.columns([4, 1])
                    row_l.write(
                        f"`{c['case_uuid'][:8]}…` via {c['via']}: {c.get('title', '')[:100]}"
                    )
                    if row_r.button("Open", key=c["case_uuid"]):
                        st.session_state["case_uuid"] = c["case_uuid"]
                        st.info("Case loaded — open ② Case review.")
            else:
                st.error(f"poll failed {r.status_code}: {r.text[:300]}")
        except Exception as exc:
            st.error(f"poll failed: {exc}")

st.divider()
st.caption(
    "Deterministic risk scores are triage priors, not probabilities. "
    "LLM narratives explain — they never re-score."
)
