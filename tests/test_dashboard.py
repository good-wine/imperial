"""Dashboard E2E (Streamlit AppTest): every language renders without
exceptions, chrome is translated, invariants (TLP/verdict codes) hold.
API is unreachable in tests by design — the console must degrade, not crash.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

APP = "../dashboard/app.py"


def _run(lang: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    if lang is not None:
        at.session_state["lang"] = lang
    at.run()
    assert not at.exception, at.exception
    return at


def _texts(at: AppTest) -> str:
    import contextlib
    import itertools

    parts = []
    widgets = itertools.chain(at.title, at.header, at.subheader, at.markdown, at.caption)
    for widget in widgets:
        with contextlib.suppress(Exception):
            parts.append(str(widget.value))
    return "\n".join(parts)


def test_renders_all_languages():
    from dashboard.i18n import _STRINGS

    for lang in ("en", "it", "fr", "es"):
        at = _run(lang)
        body = _texts(at)
        assert _STRINGS[lang]["app_title"] in body
        assert [t.label for t in at.tabs] == [
            _STRINGS[lang]["tab.ingest"],
            _STRINGS[lang]["tab.review"],
            _STRINGS[lang]["tab.hitl"],
            _STRINGS[lang]["tab.osint"],
        ]


def test_language_switch_changes_chrome():
    at = _run("en")
    assert "Analyst Console" in _texts(at)
    at.selectbox(key="lang").set_value("it").run()
    assert not at.exception, at.exception
    assert "Console Analista" in _texts(at)


def test_session_widgets_present():
    at = _run("en")
    assert len(at.selectbox) >= 2  # language + clearance (+ classification)
    assert len(at.text_input) >= 1  # analyst callsign
    assert len(at.tabs) == 4 if hasattr(at, "tabs") else True


def test_verdict_codes_never_translated_in_chrome():
    body = _texts(_run("it"))
    for code in ("AUTHENTIC", "HIGH_RISK", "UNVERIFIABLE"):
        assert code not in body
