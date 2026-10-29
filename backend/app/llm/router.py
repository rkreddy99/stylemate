"""Provider router.

Services import `llm` from here and never learn which provider answered. The
active provider can be swapped at runtime from the UI without a restart.
"""
from __future__ import annotations

from ..config import settings
from .gemini import FALLBACK_MODELS, Gemini, pick_best
from .ollama import LLMReply, LLMUnavailable, Ollama, extract_json  # noqa: F401


class Router:
    def __init__(self):
        self.ollama = Ollama()
        self.gemini = Gemini()

    # -- selection --------------------------------------------------------
    @property
    def active_name(self) -> str:
        p = (settings.provider or "auto").lower()
        if p == "gemini":
            return "gemini"
        if p == "ollama":
            return "ollama"
        return "gemini" if self.gemini.api_key else "ollama"

    @property
    def active(self):
        return self.gemini if self.active_name == "gemini" else self.ollama

    @property
    def vision_model_name(self) -> str:
        return (settings.gemini_vision_model if self.active_name == "gemini"
                else settings.vision_model)

    # -- pass-through -----------------------------------------------------
    def _fix_model(self, kw: dict) -> dict:
        """A caller asking for vision must not be handed the other provider's
        model id: 'qwen2.5vl:7b' means nothing to Gemini and vice versa."""
        if kw.get("images") and not kw.get("model"):
            kw["model"] = self.vision_model_name
        elif kw.get("images") and kw.get("model"):
            m = str(kw["model"])
            looks_gemini = m.startswith("gemini")
            if looks_gemini != (self.active_name == "gemini"):
                kw["model"] = self.vision_model_name
        return kw

    def chat(self, *a, **kw) -> LLMReply:
        return self.active.chat(*a, **self._fix_model(kw))

    def chat_json(self, *a, **kw):
        return self.active.chat_json(*a, **self._fix_model(kw))

    def embed(self, *a, **kw):
        return self.active.embed(*a, **kw)

    def health(self) -> dict:
        name = self.active_name
        if name == "gemini":
            h = self.gemini.health()
        else:
            h = self.ollama.health()
            h["provider"] = "ollama"
            h["hasKey"] = True
        h["activeProvider"] = name
        h["providerSetting"] = settings.provider
        h["geminiKeySet"] = bool(self.gemini.api_key)
        h["ollamaHost"] = self.ollama.host
        return h

    def health_all(self) -> dict:
        """Both providers, for the settings screen."""
        o = self.ollama.health()
        o["provider"] = "ollama"
        return {"active": self.active_name, "setting": settings.provider,
                "ollama": o, "gemini": self.gemini.health(),
                "geminiFallbackModels": FALLBACK_MODELS}

    # -- runtime configuration -------------------------------------------
    def configure(self, *, provider: str | None = None,
                  gemini_api_key: str | None = None,
                  gemini_text_model: str | None = None,
                  gemini_vision_model: str | None = None,
                  ollama_host: str | None = None,
                  text_model: str | None = None,
                  vision_model: str | None = None,
                  num_ctx: int | None = None) -> dict:
        if provider in ("ollama", "gemini", "auto"):
            settings.provider = provider
        auto_note = None
        if gemini_api_key is not None:
            key = gemini_api_key.strip()
            settings.gemini_api_key = key
            self.gemini.api_key = key
            if key:
                # A new key: ask Google what it can actually run and select the
                # newest usable model straight away, so the shopper does not
                # have to know which one to pick.
                info = self.gemini.list_models()
                if info.get("reachable"):
                    best = info.get("best") or pick_best(info.get("text") or [])
                    if best and not gemini_text_model:
                        settings.gemini_text_model = best
                        self.gemini.text_model = best
                    bestv = pick_best(info.get("vision") or [])
                    if bestv and not gemini_vision_model:
                        settings.gemini_vision_model = bestv
                        self.gemini.vision_model = bestv
                    embeds = info.get("embed") or []
                    if embeds:
                        settings.gemini_embed_model = embeds[0]
                        self.gemini.embed_model = embeds[0]
                    auto_note = (f"Key accepted. Selected {best} for search and "
                                 f"{bestv} for photo recognition - the newest "
                                 f"models this key can use.")
                    # A working key should actually be used.
                    if settings.provider == "auto" or not settings.provider:
                        settings.provider = "auto"
        if gemini_text_model:
            settings.gemini_text_model = gemini_text_model
            self.gemini.text_model = gemini_text_model
        if gemini_vision_model:
            settings.gemini_vision_model = gemini_vision_model
            self.gemini.vision_model = gemini_vision_model
        if ollama_host:
            settings.ollama_host = ollama_host.rstrip("/")
            self.ollama.host = settings.ollama_host
        if text_model:
            settings.text_model = text_model
        if vision_model:
            settings.vision_model = vision_model
        if num_ctx and 1024 <= int(num_ctx) <= 131072:
            settings.num_ctx = int(num_ctx)
        out = self.health_all()
        if auto_note:
            out["autoSelected"] = auto_note
        return out


llm = Router()
