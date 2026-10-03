"""i18n contract tests (no Streamlit needed — pure `dashboard/i18n.py`).

UI chrome is translated; TLP markings, verdict codes, hashes and API field
names are invariant by design. Any new string added to English without
IT/FR/ES translations fails here before it can ship a half-localized UI.
"""

from __future__ import annotations

import re

from dashboard.i18n import _STRINGS, DEFAULT_LANG, LANGS, current_lang, t

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::[^}]*)?\}")


def test_languages_declared():
    assert set(LANGS) == {"en", "it", "fr", "es"}
    assert set(_STRINGS) == {"en", "it", "fr", "es"}


def test_key_parity():
    en_keys = set(_STRINGS["en"])
    assert len(en_keys) > 80  # translation surface is covered, not stubbed
    for lang in ("it", "fr", "es"):
        missing = en_keys - set(_STRINGS[lang])
        extra = set(_STRINGS[lang]) - en_keys
        assert not missing, f"{lang} missing keys: {sorted(missing)}"
        assert not extra, f"{lang} extra keys: {sorted(extra)}"


def test_placeholder_consistency():
    for key in _STRINGS["en"]:
        expected = set(_PLACEHOLDER.findall(_STRINGS["en"][key]))
        for lang in ("it", "fr", "es"):
            got = set(_PLACEHOLDER.findall(_STRINGS[lang][key]))
            assert got == expected, f"{lang}.{key}: {sorted(got)} != {sorted(expected)}"


def test_no_empty_translations():
    for lang, table in _STRINGS.items():
        for key, value in table.items():
            assert value.strip(), f"{lang}.{key} is empty"


def test_fallback_and_format():
    assert t("xx", "sidebar.session") == _STRINGS["en"]["sidebar.session"]
    assert t("it", "no.such.key") == "no.such.key"
    assert "42" in t("en", "review.export_failed", e=42)
    assert t("it", "ingest.meta", kb=1.5, name="a.png", mime="image/png")
    # Unknown language falls back to English, never raises.
    assert t("de", "tab.ingest") == _STRINGS["en"]["tab.ingest"]


def test_invariants_not_translated():
    blob = "\n".join(v for table in _STRINGS.values() for v in table.values())
    # TLP appears as a UI label everywhere; verdict codes must never appear
    # as UI prose (values come from the API and stay canonical).
    assert "TLP" in blob
    for code in ("AUTHENTIC", "LOW_RISK", "UNVERIFIABLE", "SUSPICIOUS", "HIGH_RISK"):
        assert code not in blob


def test_current_lang_resolution(monkeypatch):
    assert current_lang("it") == "it"
    assert current_lang("de") == DEFAULT_LANG
    assert current_lang(None) == DEFAULT_LANG
    monkeypatch.setenv("DASHBOARD_LANG", "fr")
    import importlib

    from dashboard import i18n

    importlib.reload(i18n)
    try:
        assert i18n.DEFAULT_LANG == "fr"
        assert i18n.current_lang(None) == "fr"
    finally:
        monkeypatch.delenv("DASHBOARD_LANG")
        importlib.reload(i18n)
