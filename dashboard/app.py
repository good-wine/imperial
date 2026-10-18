"""Analyst dashboard (Streamlit) — HITL verification console, intel UX.

Run: `streamlit run dashboard/app.py` (needs `pip install streamlit requests`).
Talks only to the FastAPI (`API_BASE`); the air-gap boundary is unchanged.
Every panel degrades with an explicit message when its backend leg is
unavailable — never a silent blank panel.

Layout: TLP session banner → sidebar (health, clearance, metrics) →
wizard tabs Ingest → Review (per-leg sub-tabs) → HITL → OSINT.

Internationalization: EN/IT/FR/ES via `dashboard/i18n.py` (zero extra
dependencies). Language selector in the sidebar; `DASHBOARD_LANG` sets the
deployment default. TLP markings, verdict codes and hashes are never
translated — only UI chrome is.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.i18n import LANGS, current_lang, t

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8088")
#: Optional bearer for deployments with OIDC clearance binding enabled
#: (`OIDC_BIND_CLEARANCE=1`): without it the console is capped at TLP:CLEAR.
API_BEARER = os.getenv("DASHBOARD_API_BEARER", "")

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
    # Verdict codes are never translated.
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


def _stepper(status: str, lang: str) -> None:
    idx = _STATUS_STEPS.index(status) if status in _STATUS_STEPS else 0
    cols = st.columns(3)
    names = [t(lang, "step.ingested"), t(lang, "step.analyzing"), t(lang, "step.done")]
    for i, name in enumerate(names):
        cls = (
            "step-done"
            if (i <= idx and status == "done") or i < idx
            else ("step-done" if i == idx else "step-todo")
        )
        mark = "●" if (i < idx or status == "done") else ("◐" if i == idx else "○")
        cols[i].markdown(f'<span class="{cls}">{mark} {name}</span>', unsafe_allow_html=True)
    if status == "failed":
        st.error(t(lang, "step.failed"))


if "lang" not in st.session_state:
    st.session_state["lang"] = current_lang(None)
lang: str = st.session_state["lang"]
if lang not in LANGS:
    lang = "en"

st.set_page_config(page_title=t(lang, "page_title"), layout="wide")
_css()
st.title(t(lang, "app_title"))


# ---------------------------------------------------------------- sidebar: session + health


def _api_health(base: str) -> tuple[bool, str]:
    try:
        r = requests.get(base + "/health", timeout=5)
        if r.status_code == 200:
            body = r.json()
            dev = (body.get("compute") or {}).get("torch_device", "?")
            return True, t(lang, "api.ok", dev=dev)
        return False, t(lang, "api.http", code=r.status_code)
    except Exception as exc:
        return False, t(lang, "api.down", err=str(exc))[:90]


with st.sidebar:
    st.header(t(lang, "sidebar.session"))
    st.selectbox(
        t(lang, "sidebar.language"),
        options=list(LANGS),
        format_func=lambda c: LANGS[c],
        key="lang",
    )
    lang = st.session_state["lang"]
    clearance = st.selectbox(
        t(lang, "sidebar.clearance"), ["TLP:CLEAR", "TLP:GREEN", "TLP:AMBER", "TLP:RED"]
    )
    analyst = st.text_input(t(lang, "sidebar.analyst"), "analyst")
    api_base = st.text_input(t(lang, "sidebar.api_base"), API_BASE)
    ok, msg = _api_health(api_base)
    st.markdown("🟢 " + msg if ok else "🔴 " + msg)
    with st.expander(t(lang, "sidebar.metrics")):
        try:
            m = requests.get(api_base + "/metrics", timeout=8).json().get("metrics", {})
            st.write(dict(m))
        except Exception:
            st.caption(t(lang, "sidebar.metrics_unavailable"))
    st.caption(t(lang, "sidebar.caption"))

_tlp_banner(clearance, t(lang, "sidebar.session_banner"))


def _headers() -> dict:
    """Request headers: declared clearance, plus the bearer when configured.

    The server caps the declared clearance by the token's claim when
    `OIDC_BIND_CLEARANCE=1`, so sending the bearer is what lets a cleared
    analyst read above TLP:CLEAR in that mode.
    """
    h = {"X-TLP": clearance}
    if API_BEARER:
        h["Authorization"] = f"Bearer {API_BEARER}"
    return h


def _get(path: str, **kw):
    try:
        return requests.get(api_base + path, headers=_headers(), timeout=30, **kw)
    except Exception as exc:
        st.error(t(lang, "api.unreachable", err=exc))
        return None


def _fetch_case(cuuid: str):
    r = _get(f"/api/v1/cases/{cuuid}", params={"clearance": clearance})
    if r is None:
        return None
    if r.status_code == 200:
        return r.json()
    if r.status_code == 403:
        st.warning(t(lang, "review.fetch_403"))
    else:
        st.error(t(lang, "review.fetch_failed", code=r.status_code, text=r.text[:300]))
    return None


# ---------------------------------------------------------------- 1 · ingest

tab_up, tab_case, tab_hitl, tab_osint = st.tabs(
    [
        t(lang, "tab.ingest"),
        t(lang, "tab.review"),
        t(lang, "tab.hitl"),
        t(lang, "tab.osint"),
    ]
)

with tab_up:
    left, right = st.columns([3, 2])
    with left:
        st.subheader(t(lang, "ingest.file"))
        up = st.file_uploader(
            t(lang, "ingest.uploader"),
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
                st.caption(
                    t(
                        lang,
                        "ingest.meta",
                        kb=len(raw) / 1024,
                        name=up.name,
                        mime=up.type or t(lang, "ingest.unknown_type"),
                    )
                )
            with meta_r:
                if (up.type or "").startswith("image/"):
                    try:
                        from PIL import Image as _PI

                        img = _PI.open(io.BytesIO(raw)).convert("RGB")
                        img.thumbnail((360, 360))
                        st.image(
                            img,
                            caption=t(lang, "ingest.preview_caption", w=img.size[0], h=img.size[1]),
                        )
                    except Exception:
                        st.caption(t(lang, "ingest.preview_unavailable"))
                elif (up.type or "").startswith("audio/"):
                    st.audio(raw)
                else:
                    st.caption(t(lang, "ingest.preview_limits"))
    with right:
        st.subheader(t(lang, "ingest.handling"))
        tlp = st.selectbox(
            t(lang, "ingest.classification"),
            ["TLP:CLEAR", "TLP:GREEN", "TLP:AMBER", "TLP:RED"],
            index=2,
            help=t(lang, "ingest.classification_help"),
        )
        _tlp_banner(tlp, t(lang, "ingest.selected_classification"))
        with st.expander(t(lang, "ingest.geoint")):
            lat = st.text_input(t(lang, "ingest.lat"), "")
            lon = st.text_input(t(lang, "ingest.lon"), "")
            ts = st.text_input(t(lang, "ingest.ts"), "")
        submit = st.button(
            t(lang, "ingest.submit"), type="primary", disabled=up is None, use_container_width=True
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
                headers=_headers(),
                timeout=120,
            )
            if r.status_code == 200:
                body = r.json()
                st.session_state["case_uuid"] = body["case_uuid"]
                st.success(
                    t(
                        lang,
                        "ingest.success",
                        uuid=body["case_uuid"][:8],
                        task=str(body["task_id"])[:8],
                    )
                )
            elif r.status_code == 429:
                st.warning(t(lang, "ingest.rate_limited"))
            else:
                st.error(t(lang, "ingest.failed", code=r.status_code, text=r.text[:400]))
        except Exception as exc:
            st.error(t(lang, "ingest.submit_failed", err=exc))

# ---------------------------------------------------------------- 2 · review

with tab_case:
    st.subheader(t(lang, "review.title"))
    c1, c2 = st.columns([4, 1])
    with c1:
        cuuid = st.text_input(
            t(lang, "review.title"),
            st.session_state.get("case_uuid", ""),
            label_visibility="collapsed",
            placeholder=t(lang, "review.uuid_placeholder"),
        )
    with c2:
        if st.button(t(lang, "review.open"), use_container_width=True) and cuuid:
            st.session_state["case_uuid"] = cuuid
    cuuid = st.session_state.get("case_uuid", "")
    if not cuuid:
        st.info(t(lang, "review.empty"))
    else:
        case = _fetch_case(cuuid)
        if case:
            _tlp_banner(
                case.get("tlp", "TLP:CLEAR"),
                f"case {cuuid[:8]}… · `{case.get('sha256', '')[:16]}…`",
            )
            _stepper(case.get("status", "queued"), lang)
            if case.get("status") not in ("done", "failed"):
                bar = st.progress(0.15, text=t(lang, "review.polling"))
                for i in range(20):
                    time.sleep(4)
                    case = _fetch_case(cuuid) or case
                    bar.progress(min(0.15 + 0.04 * (i + 1), 0.95))
                    if case.get("status") in ("done", "failed"):
                        break
                else:
                    st.warning(t(lang, "review.still_running"))
                bar.empty()
                _stepper(case.get("status", "queued"), lang)
            res = case.get("result") or {}
            if not res:
                st.info(t(lang, "review.no_findings"))
            else:
                risk = int(res.get("risk_score", 0))
                _verdict_banner(str(res.get("verdict", "")), risk)
                k1, k2, k3, k4 = st.columns([1, 1, 1, 2])
                with k1:
                    _risk_gauge(risk)
                with k2:
                    st.metric(t(lang, "review.confidence"), res.get("analytic_confidence", "?"))
                    st.metric(t(lang, "review.llm_backend"), res.get("llm_backend", "?"))
                with k3:
                    st.metric(t(lang, "review.schema"), res.get("schema_version", "?"))
                    st.metric(t(lang, "review.tlp"), res.get("tlp", "?"))
                with k4:
                    st.markdown(f"**{t(lang, 'review.exec_summary')}**")
                    st.write(res.get("executive_summary", ""))
                nt, fo, au, bi, ge, cx, ex = st.tabs(
                    [
                        t(lang, "review.sub_narrative"),
                        t(lang, "review.sub_forensics"),
                        t(lang, "review.sub_audio"),
                        t(lang, "review.sub_bio"),
                        t(lang, "review.sub_geoint"),
                        t(lang, "review.sub_context"),
                        t(lang, "review.sub_export"),
                    ]
                )
                with nt:
                    st.write(res.get("narrative", t(lang, "review.narrative_withheld")))
                    hyps = res.get("hypotheses", [])
                    if hyps:
                        st.markdown(f"**{t(lang, 'review.ach')}**")
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
                    with st.expander(t(lang, "review.findings")):
                        for f in res.get("findings", []):
                            st.write(f"- {f}")
                        for x in res.get("recommendations", []):
                            st.write(f"+ {x}")

                    # EU AI Act Art. 50 transparency block. Read the top-level
                    # payload the case endpoint serves (clearance-scoped), not
                    # `result`: at AMBER/RED `result.ai_act50` is absent, and a
                    # truthy guard on it fabricated a reassuring "no marking".
                    ai_act = case.get("ai_act50") or {}
                    if ai_act.get("available"):
                        with st.expander(t(lang, "review.act50_title")):
                            a = ai_act
                            st.write(
                                t(
                                    lang,
                                    "review.act50_marking",
                                    m=a.get("machine_readable_marking", "ABSENT"),
                                    s=a.get("marking_standard")
                                    or t(lang, "review.act50_no_standard"),
                                )
                            )
                            if a.get("disclosure_required") is True:
                                disclosure_key = "review.act50_yes"
                            elif a.get("disclosure_required") is False:
                                disclosure_key = "review.act50_no"
                            else:
                                disclosure_key = "review.act50_undetermined"
                            st.write(
                                t(
                                    lang,
                                    "review.act50_disclosure",
                                    r=t(lang, disclosure_key),
                                )
                            )
                            if a.get("label_text"):
                                st.write(t(lang, "review.act50_label", l=a["label_text"]))
                            if a.get("restricted"):
                                # Withheld fields must read as withheld, not as
                                # "assessed: none" (the fabricated-default bug).
                                st.caption(t(lang, "review.act50_restricted"))
                            else:
                                if a.get("deepfake") is None:
                                    det_label = t(lang, "review.act50_indeterminate")
                                elif a.get("deepfake"):
                                    det_label = t(lang, "review.act50_deepfake")
                                else:
                                    det_label = t(lang, "review.act50_not_deepfake")
                                st.write(
                                    t(
                                        lang,
                                        "review.act50_determination",
                                        d=det_label,
                                        c=a.get("deepfake_confidence", "Indeterminate"),
                                    )
                                )
                                st.write(
                                    t(
                                        lang,
                                        "review.act50_subjects",
                                        s=", ".join(a.get("depicted_subject", []))
                                        or t(lang, "review.act50_none"),
                                    )
                                )
                                st.write(
                                    t(
                                        lang,
                                        "review.act50_audience",
                                        r=a.get("false_appearance_risk", "unknown"),
                                    )
                                )
                                if a.get("exemption"):
                                    st.info(t(lang, "review.act50_exemption", e=a["exemption"]))
                            badge = a.get("disclosure_badge_svg")
                            if badge:
                                st.markdown(badge, unsafe_allow_html=True)
                                st.caption(t(lang, "review.act50_badge_note"))

                with fo:
                    fr = res.get("forensic") or {}
                    st.caption(
                        t(
                            lang,
                            "review.forensic_caption",
                            a=fr.get("aggregate_anomaly"),
                            n=fr.get("num_frames_analyzed"),
                            d=fr.get("device_used"),
                            o=fr.get("onnx_used"),
                            t=fr.get("temporal_consistency"),
                        )
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
                            t(
                                lang,
                                "review.mp4_caption",
                                n=len(mp.get("atoms", [])),
                                f=mp.get("faststart"),
                                h=mp.get("double_compression_hint"),
                            )
                        )
                        for a in mp.get("anomalies", []):
                            st.warning(a)
                    wm = (res.get("multimodal") or {}).get("watermark") or {}
                    if wm.get("metadata_hits") or wm.get("ai_generated"):
                        st.error(
                            t(
                                lang,
                                "review.ai_generated",
                                v=wm.get("ai_generated"),
                                g=wm.get("generator"),
                                s=wm.get("synthid"),
                            )
                        )
                        for h in wm.get("metadata_hits", [])[:5]:
                            st.code(h)
                with au:
                    au = (res.get("multimodal") or {}).get("audio") or {}
                    if au.get("available"):
                        st.metric(t(lang, "review.audio_vocoder"), au.get("vocoder_anomaly"))
                        st.metric(t(lang, "review.audio_pitch"), au.get("pitch_instability"))
                        for n in au.get("spectral_notes", []):
                            st.caption(n)
                        for w in au.get("warnings", []):
                            st.warning(w)
                    else:
                        st.caption("; ".join(au.get("warnings", [t(lang, "review.audio_default")])))
                    ls = (res.get("multimodal") or {}).get("lipsync") or {}
                    if ls.get("available"):
                        st.metric(
                            "Lip-sync",
                            t(lang, "review.lipsync", v=ls.get("verdict"), r=ls.get("correlation")),
                        )
                    sp = (res.get("multimodal") or {}).get("speech") or {}
                    if sp.get("available"):
                        st.markdown(
                            t(
                                lang,
                                "review.transcript",
                                lng=sp.get("language", "?"),
                                p=sp.get("language_probability"),
                            )
                        )
                        st.write(sp.get("text", ""))
                        if sp.get("english_gloss"):
                            with st.expander(t(lang, "review.translation")):
                                st.write(sp["english_gloss"])
                    elif sp.get("errors"):
                        st.caption("; ".join(sp["errors"][:1]))
                with bi:
                    rp = (res.get("multimodal") or {}).get("rppg") or {}
                    if rp.get("available"):
                        st.metric(t(lang, "review.bpm"), f"{rp.get('heart_rate_bpm')} bpm")
                        st.metric(
                            t(lang, "review.liveness"),
                            f"{rp.get('liveness')} (q={rp.get('signal_quality')})",
                        )
                        if rp.get("waveform"):
                            st.line_chart(rp["waveform"])
                        for w in rp.get("warnings", []):
                            st.warning(w)
                    else:
                        st.caption(
                            "; ".join(
                                rp.get("errors")
                                or rp.get("warnings")
                                or [t(lang, "review.rppg_needs")]
                            )
                        )
                with ge:
                    geo = (res.get("multimodal") or {}).get("geoint") or {}
                    if geo.get("available"):
                        g1, g2, g3 = st.columns(3)
                        g1.metric(
                            t(lang, "review.sun"),
                            f"{geo.get('solar_elevation_deg')}° / {geo.get('solar_azimuth_deg')}°",
                        )
                        g2.metric(t(lang, "review.shadow_err"), f"{geo.get('angular_error_deg')}°")
                        g3.metric(t(lang, "review.consistent"), str(geo.get("consistent")))
                        if geo.get("latitude") is not None:
                            st.map(data=[{"lat": geo["latitude"], "lon": geo["longitude"]}])
                        for w in geo.get("warnings", []):
                            st.warning(w)
                    else:
                        st.caption(t(lang, "review.geoint_needs"))
                with cx:
                    ctx0 = res.get("context") or {}
                    clip_key = "review.clip_yes" if ctx0.get("clip_available") else "review.clip_no"
                    st.caption(
                        f"backend {ctx0.get('backend', '?')} · "
                        f"phash {str(ctx0.get('query_phash', ''))[:12]}… · "
                        f"clip={t(lang, clip_key)}"
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
                            st.caption(t(lang, "review.no_corr"))
                    else:
                        st.caption(t(lang, "review.no_corr_backend"))
                    for c in res.get("citations", [])[:5]:
                        st.caption(f"ref: {c}")
                with ex:
                    if st.button(t(lang, "review.export"), type="primary"):
                        r3 = _get(f"/api/v1/export/stix/{cuuid}", params={"clearance": clearance})
                        if r3 and r3.status_code == 200:
                            st.download_button(
                                t(lang, "review.download"),
                                json.dumps(r3.json(), indent=2),
                                file_name=f"{cuuid}.stix.json",
                                mime="application/json",
                            )
                            st.caption(t(lang, "review.export_caption"))
                        else:
                            st.error(
                                t(
                                    lang,
                                    "review.export_failed",
                                    e=r3.status_code if r3 else t(lang, "api.unreachable", err=""),
                                )
                            )
                    rd = (res.get("multimodal") or {}).get("redaction") or {}
                    if rd.get("redacted_path"):
                        st.success(
                            t(
                                lang,
                                "review.redacted",
                                p=rd["redacted_path"],
                                f=rd.get("faces_blurred"),
                                pl=rd.get("plates_blurred"),
                            )
                        )
            with st.expander(t(lang, "review.audit")):
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
                    st.caption(t(lang, "review.audit_empty"))

# ---------------------------------------------------------------- 3 · HITL

with tab_hitl:
    st.subheader(t(lang, "hitl.title"))
    cuuid = st.session_state.get("case_uuid", "")
    if not cuuid:
        st.info(t(lang, "hitl.empty"))
    else:
        case = _fetch_case(cuuid)
        if case and case.get("result"):
            st.caption(
                t(
                    lang,
                    "hitl.current",
                    v=case["result"].get("verdict"),
                    r=case["result"].get("risk_score"),
                )
            )
        label = st.radio(
            t(lang, "hitl.correction"),
            ["CONFIRM", "FP", "FN"],
            format_func=lambda x: {
                "CONFIRM": t(lang, "hitl.confirm"),
                "FP": t(lang, "hitl.fp"),
                "FN": t(lang, "hitl.fn"),
            }[x],
        )
        finding_ref = st.text_input(
            t(lang, "hitl.finding_ref"), placeholder=t(lang, "hitl.finding_ph")
        )
        notes = st.text_area(
            t(lang, "hitl.notes"),
            placeholder=t(lang, "hitl.notes_ph"),
        )
        if st.button(t(lang, "hitl.submit"), type="primary"):
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
                    st.success(t(lang, "hitl.success", id=r.json()["feedback_id"]))
                else:
                    st.error(t(lang, "hitl.failed", code=r.status_code, text=r.text[:300]))
            except Exception as exc:
                st.error(t(lang, "hitl.submit_failed", err=exc))

# ---------------------------------------------------------------- 4 · OSINT

with tab_osint:
    st.subheader(t(lang, "osint.title"))
    st.caption(t(lang, "osint.caption"))
    if st.button(t(lang, "osint.run"), type="primary"):
        try:
            with st.spinner(t(lang, "osint.polling")):
                r = requests.post(api_base + "/api/v1/osint/poll", timeout=300)
            if r.status_code == 200:
                body = r.json()
                st.success(t(lang, "osint.success", n=body["created"]))
                for c in body.get("cases", [])[:20]:
                    row_l, row_r = st.columns([4, 1])
                    row_l.write(
                        f"`{c['case_uuid'][:8]}…` via {c['via']}: {c.get('title', '')[:100]}"
                    )
                    if row_r.button(t(lang, "osint.open"), key=c["case_uuid"]):
                        st.session_state["case_uuid"] = c["case_uuid"]
                        st.info(t(lang, "osint.loaded"))
            else:
                st.error(t(lang, "osint.failed", code=r.status_code, text=r.text[:300]))
        except Exception as exc:
            st.error(t(lang, "osint.poll_failed", err=exc))

st.divider()
st.caption(t(lang, "footer"))
