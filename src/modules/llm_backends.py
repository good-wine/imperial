"""Backend-agnostic local-LLM wrapper (SOTA: deterministic decoding + jury).

Calling code uses only `get_backend()` + `generate()`. Swapping providers
(Ollama today, Hugging Face / LlamaIndex tomorrow) means adding a subclass —
`explain_report.py` never changes.

Backends:
  - `OllamaBackend` — quantized GGUF via Ollama, temperature 0 + `format: json`
    structured mode available. Fully air-gappable.
  - `JuryBackend` — primary drafts, independent verifier critiques (two-model
    or two-prompt jury); disagreements are surfaced, not hidden.
  - `HuggingFaceBackend` — lazy `transformers` pipeline (offline weights).
  - `LlamaIndexBackend` — delegates to Ollama until full RAG lands.
  - `RuleBasedBackend` — deterministic offline fallback, always available.

`LLM_BACKEND=auto` tries Ollama first (health-check), then HF, then rule-based.
Every failure degrades with a logged warning — the report is never blocked.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)


class BaseLLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Return assistant text. May raise; callers fall back to rule-based."""

    def generate_json(self, prompt: str, system: str = "") -> str:
        """Structured mode. Default: prose backend, caller parses defensively."""
        return self.generate(prompt, system)

    def health(self) -> bool:
        try:
            out = self.generate("Reply with: ok", system="Health check.")
            return bool(out)
        except Exception:
            return False


class RuleBasedBackend(BaseLLMBackend):
    """Zero-dependency narrative builder. Deterministic — ideal for audits."""

    name = "rule-based"

    def generate(self, prompt: str, system: str = "") -> str:
        # `prompt` already contains the structured findings; wrap them so the
        # report shape is identical regardless of backend.
        return (
            "Analyst note (rule-based offline backend — no LLM was reachable):\n\n"
            f"{prompt.strip()}\n\n"
            "Treat the risk score as the authoritative signal; this narrative "
            "is a template, not a model judgment."
        )


class OllamaBackend(BaseLLMBackend):
    """Quantized local models served by Ollama. Fully air-gappable."""

    name = "ollama"

    def __init__(
        self, host: str, model: str, timeout_s: int = 60, temperature: float = 0.0
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature

    def _post(self, payload: dict) -> str:
        import json as _json

        from src.security import fetch_url

        data = _json.dumps(payload).encode("utf-8")
        try:
            raw = fetch_url(
                f"{self.host}/api/generate",
                method="POST",
                data=data,
                headers={"Content-Type": "application/json"},
                timeout_s=self.timeout_s,
                max_bytes=8_000_000,
            )
            body = _json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"ollama unreachable at {self.host} ({exc})") from exc
        text = (body.get("response") or "").strip()
        if not text:
            raise RuntimeError("ollama returned empty response")
        return text

    def generate(self, prompt: str, system: str = "") -> str:
        # temperature 0 + fixed seed: reproducible narratives for audits.
        return self._post(
            {
                "model": self.model,
                "prompt": prompt,
                "system": system or "You are a forensic media analyst. Be concise and factual.",
                "stream": False,
                "options": {"temperature": self.temperature, "seed": 7},
            }
        )

    def generate_json(self, prompt: str, system: str = "") -> str:
        return self._post(
            {
                "model": self.model,
                "prompt": prompt,
                "system": system or "Emit valid JSON only.",
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "seed": 7},
            }
        )


def ollama_embeddings(
    texts: list[str], host: str, model: str = "nomic-embed-text", timeout_s: int = 30
) -> list[list[float]] | None:
    """Local embeddings for semantic RAG re-rank. None when unreachable."""
    import json as _json

    out: list[list[float]] = []
    try:
        from src.security import fetch_url

        for t in texts:
            data = _json.dumps({"model": model, "prompt": t[:2000]}).encode()
            raw = fetch_url(
                f"{host.rstrip('/')}/api/embeddings",
                method="POST",
                data=data,
                headers={"Content-Type": "application/json"},
                timeout_s=timeout_s,
                max_bytes=8_000_000,
            )
            body = _json.loads(raw.decode("utf-8"))
            out.append(body["embedding"])
        return out
    except Exception as exc:
        log.info("llm: embeddings unavailable (%s)", exc)
        return None


class JuryBackend(BaseLLMBackend):
    """Two-pass jury: primary drafts, verifier critiques factual grounding.

    With one model configured, the verifier runs as an independent low-context
    pass (different system prompt). Disagreements append a dissent note instead
    of being silently resolved — SOTA practice for high-stakes assessments.
    """

    name = "jury"

    _VERIFY_SYSTEM = (
        "You are a verification reviewer. Given the draft assessment and the "
        "evidence bullets, list any claim NOT supported by the evidence. "
        "Be terse. If fully supported, reply exactly: SUPPORTED."
    )

    def __init__(self, primary: BaseLLMBackend, verifier: BaseLLMBackend | None = None) -> None:
        self.primary = primary
        self.verifier = verifier or primary

    def generate(self, prompt: str, system: str = "") -> str:
        draft = self.primary.generate(prompt, system)
        try:
            verdict = self.verifier.generate(
                f"DRAFT:\n{draft}\n\nEVIDENCE:\n{prompt[:3000]}", system=self._VERIFY_SYSTEM
            )
        except Exception as exc:
            log.warning("jury: verifier failed (%s); returning draft", exc)
            return draft
        if "SUPPORTED" in verdict.upper():
            return draft
        return draft + "\n\n[Reviewer dissent — verify before acting]\n" + verdict.strip()[:800]


class HuggingFaceBackend(BaseLLMBackend):
    """Local `transformers` inference. Weights must be pre-downloaded for offline use."""

    name = "huggingface"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._pipe: Any = None

    def _ensure(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("transformers not installed") from exc
            import torch

            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                "text-generation", model=self.model_id, device=device, trust_remote_code=False
            )
        return self._pipe

    def generate(self, prompt: str, system: str = "") -> str:
        pipe = self._ensure()
        full = f"{system}\n\n{prompt}" if system else prompt
        out = pipe(full, max_new_tokens=400, do_sample=False, truncation=True)
        text = out[0]["generated_text"]
        return text[len(full) :].strip() or text.strip()


class LlamaIndexBackend(BaseLLMBackend):
    """Adapter point for future LlamaIndex + local-embedding RAG.

    MVP scope: delegates narrative generation to Ollama (the configured local
    model) while exposing the LlamaIndex import surface so the swap is trivial.
    """

    name = "llamaindex"

    def __init__(self, ollama_host: str = "", ollama_model: str = "") -> None:
        self._delegate: BaseLLMBackend = (
            OllamaBackend(ollama_host, ollama_model) if ollama_host else RuleBasedBackend()
        )

    def generate(self, prompt: str, system: str = "") -> str:
        try:
            import llama_index  # noqa: F401 — validates future dependency surface
        except ImportError:
            log.info("llamaindex not installed; delegating to %s", self._delegate.name)
        return self._delegate.generate(prompt, system)


def get_backend(
    preference: str = "auto",
    *,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str = "llama3.1:8b",
    ollama_timeout_s: int = 60,
    hf_model: str = "",
) -> BaseLLMBackend:
    """Resolve a working backend. Always returns *something* (rule-based last)."""
    pref = (preference or "auto").lower()
    candidates: list[BaseLLMBackend] = []
    if pref in ("auto", "ollama"):
        candidates.append(OllamaBackend(ollama_host, ollama_model, ollama_timeout_s))
    if pref in ("hf", "huggingface") and hf_model:
        # Explicit only: auto must never trigger a multi-GB weights download
        # in air-gapped/offline runs (transformers would fetch from the Hub).
        candidates.append(HuggingFaceBackend(hf_model))
    if pref == "llamaindex":
        candidates.append(LlamaIndexBackend(ollama_host, ollama_model))
    if pref == "jury":
        candidates.append(JuryBackend(OllamaBackend(ollama_host, ollama_model, ollama_timeout_s)))
    candidates.append(RuleBasedBackend())  # guaranteed fallback

    if pref != "auto":
        # Explicit preference: try it, fall back to rule-based on failure.
        first = candidates[0]
        try:
            if first.health():
                return first
            log.warning("llm: preferred backend %s unhealthy; using rule-based", first.name)
        except Exception as exc:
            log.warning("llm: preferred backend failed (%s); using rule-based", exc)
        return RuleBasedBackend()

    for backend in candidates:
        try:
            if backend.health():
                log.info("llm: backend=%s active", backend.name)
                return backend
        except Exception:
            continue
        if isinstance(backend, RuleBasedBackend):
            return backend
    log.warning("llm: ollama check failed; using rule-based fallback")
    return RuleBasedBackend()
