"""Google Gemini provider.

Same surface as the Ollama client (chat / chat_json / embed / health) so every
service keeps working unchanged whichever provider is selected.

The API key lives on the server. It is accepted once over POST /api/settings/llm
or read from GEMINI_API_KEY, held in memory, never written to disk, never
logged, and never returned by any endpoint - responses only ever say whether a
key is present.
"""
from __future__ import annotations

import base64
import time

import httpx

from ..config import settings
from .ollama import LLMReply, LLMUnavailable, extract_json

BASE = "https://generativelanguage.googleapis.com/v1beta"

# Known-good model ids, used when the live list cannot be fetched. The live
# list from /api/models is always preferred so this never goes stale.
FALLBACK_MODELS = {
    "text": [
        "gemini-4-argon", "gemini-3.8-flash", "gemini-3.6-flash",
        "gemini-3.5-flash-lite", "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite", "gemini-3-flash-preview",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    ],
    "vision": [
        "gemini-4-argon", "gemini-3.8-flash", "gemini-3.6-flash",
        "gemini-3.5-flash-lite", "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite", "gemini-3-flash-preview",
        "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    ],
    "embed": ["gemini-embedding-2-preview", "gemini-embedding-001"],
}

# Only ever a stop-gap: the live list from /api/models is what the UI shows
# whenever a key is set, so new Google releases appear without a code change.

# Model families we never offer for chat, whatever the live list returns.
_EXCLUDE = ("tts", "-live", "transcribe", "-image", "lyria", "veo", "omni",
            "robotics", "deep-research", "antigravity", "aqa", "imagen")


def _version_key(model_id: str) -> tuple:
    """Sort newest-first. 'gemini-3.8-flash' must outrank 'gemini-2.5-pro'."""
    import re
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", model_id)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2)) if (m and m.group(2)) else 0
    # Prefer general models over previews, and flash over lite for speed/cost.
    preview = 1 if "preview" in model_id else 0
    lite = 1 if "lite" in model_id else 0
    return (-major, -minor, preview, lite, model_id)


def pick_best(models: list[str], prefer: str = "flash") -> str | None:
    """The model to select by default: newest generation, then `prefer`."""
    usable = [m for m in models if not any(x in m for x in _EXCLUDE)]
    if not usable:
        return None
    favoured = [m for m in usable if prefer in m] or usable
    return sorted(favoured, key=_version_key)[0]

# Every current Gemini text model is multimodal, so the vision list is simply
# the text list minus anything embedding-only.
_EMBED_HINTS = ("embedding", "embed")


class Gemini:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.text_model = settings.gemini_text_model
        self.vision_model = settings.gemini_vision_model
        self.embed_model = settings.gemini_embed_model

    # -- plumbing ---------------------------------------------------------
    def _require_key(self) -> str:
        if not self.api_key:
            raise LLMUnavailable(
                "No Gemini API key is set. Add it under Agent model, or set "
                "GEMINI_API_KEY before starting the server.")
        return self.api_key

    def _post(self, path: str, payload: dict, timeout: float | None = None,
              model: str = "") -> dict:
        key = self._require_key()
        try:
            with httpx.Client(timeout=timeout or settings.llm_timeout) as c:
                r = c.post(f"{BASE}/{path}", json=payload,
                           headers={"x-goog-api-key": key})
                r.raise_for_status()
                return r.json()
        except httpx.TimeoutException as e:
            raise LLMUnavailable("Gemini did not answer in time.") from e
        except httpx.ProxyError as e:
            raise LLMUnavailable(
                f"A network proxy refused the connection to Google's API "
                f"({e}). Check HTTPS_PROXY / corporate proxy settings.") from e
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            body = e.response.text[:300]
            if code in (401, 403):
                raise LLMUnavailable(
                    "Gemini rejected the API key. Check the key is correct and "
                    "that the Generative Language API is enabled for it.") from e
            if code == 429:
                raise LLMUnavailable(
                    "Gemini rate limit reached for this key. Wait a moment or "
                    "switch to a lighter model such as gemini-2.5-flash.") from e
            if code == 404:
                raise LLMUnavailable(
                    f"Gemini has no model called '{model}', or it does not "
                    "support this kind of request. Pick another from the "
                    "model list.") from e
            raise LLMUnavailable(f"Gemini returned {code}: {body}") from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(
                f"Could not reach Google's API: {e}. Check this machine's "
                "internet connection and any proxy settings.") from e

    # -- health -----------------------------------------------------------
    def list_models(self) -> dict:
        """Live model list for this key, cached for five minutes.

        Every open tab polls the status endpoint every 20 seconds, and each poll
        lands here. Uncached, a hosted site with a handful of visitors would ask
        Google for the model list several times a second."""
        cached = getattr(self, "_models_cache", None)
        if cached and cached[0] == self.api_key and time.time() - cached[1] < 300:
            return cached[2]
        out = self._list_models_live()
        if out.get("reachable"):
            self._models_cache = (self.api_key, time.time(), out)
        return out

    def _list_models_live(self) -> dict:
        key = self.api_key
        if not key:
            return {"reachable": False, "models": [], **FALLBACK_MODELS,
                    "source": "fallback",
                    "error": "No API key set."}
        try:
            with httpx.Client(timeout=15) as c:
                r = c.get(f"{BASE}/models", headers={"x-goog-api-key": key})
                r.raise_for_status()
                data = r.json()
        except Exception as e:                       # noqa: BLE001
            return {"reachable": False, "models": [], **FALLBACK_MODELS,
                    "source": "fallback", "error": str(e)[:200]}
        names, text, vision, embed = [], [], [], []
        for m in data.get("models", []):
            mid = str(m.get("name", "")).split("/")[-1]
            if not mid:
                continue
            methods = m.get("supportedGenerationMethods") or []
            names.append(mid)
            if any(h in mid for h in _EMBED_HINTS):
                if "embedContent" in methods or not methods:
                    embed.append(mid)
                continue
            if "generateContent" in methods and not any(x in mid for x in _EXCLUDE):
                text.append(mid)
                vision.append(mid)
        return {"reachable": True, "models": sorted(names), "source": "live",
                "text": sorted(text, key=_version_key) or FALLBACK_MODELS["text"],
                "vision": sorted(vision, key=_version_key) or FALLBACK_MODELS["vision"],
                "embed": sorted(embed) or FALLBACK_MODELS["embed"],
                "best": pick_best(text or FALLBACK_MODELS["text"])}

    def health(self) -> dict:
        info = self.list_models()
        avail = set(info.get("models") or [])
        def ready(m):
            return bool(info.get("reachable") and (not avail or m in avail))
        return {
            "provider": "gemini",
            "reachable": info.get("reachable", False),
            "hasKey": bool(self.api_key),
            "host": BASE,
            "models": info.get("models", []),
            "modelLists": {k: info.get(k) or FALLBACK_MODELS[k]
                           for k in ("text", "vision", "embed")},
            "textModel": self.text_model,
            "textModelReady": ready(self.text_model),
            "visionModel": self.vision_model,
            "visionModelReady": ready(self.vision_model),
            "embedModel": self.embed_model,
            "embedModelReady": ready(self.embed_model),
            "error": info.get("error"),
        }

    # -- chat -------------------------------------------------------------
    def chat(self, system: str, user: str, *, images: list[bytes] | None = None,
             model: str | None = None, json_mode: bool = False,
             temperature: float = 0.2, timeout: float | None = None,
             response_schema: dict | None = None,
             max_output_tokens: int | None = None) -> LLMReply:
        mid = model or (self.vision_model if images else self.text_model)
        parts: list[dict] = [{"text": user}]
        for b in (images or []):
            parts.append({"inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(b).decode()}})
        payload: dict = {
            "contents": [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": temperature},
        }
        if json_mode or response_schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        # A schema makes the reply's shape the API's job rather than the
        # prompt's, so there is nothing to repair and no prose to pay for.
        if response_schema:
            payload["generationConfig"]["responseSchema"] = response_schema
        if max_output_tokens:
            payload["generationConfig"]["maxOutputTokens"] = int(max_output_tokens)
        t0 = time.time()
        try:
            data = self._post(f"models/{mid}:generateContent", payload, timeout,
                              model=mid)
        except LLMUnavailable as e:
            # A model that rejects the schema still answers JSON mode fine.
            if response_schema and "returned 400" in str(e):
                payload["generationConfig"].pop("responseSchema", None)
                data = self._post(f"models/{mid}:generateContent", payload,
                                  timeout, model=mid)
            else:
                raise

        cands = data.get("candidates") or []
        if not cands:
            fb = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMUnavailable(
                f"Gemini returned no answer"
                + (f" (blocked: {fb}). Try a different photo or wording."
                   if fb else ". Try again."))
        cand = cands[0]
        text = "".join(p.get("text", "")
                       for p in (cand.get("content") or {}).get("parts", []))
        if not text and cand.get("finishReason") == "MAX_TOKENS":
            raise LLMUnavailable(
                "Gemini hit its output limit before producing anything usable.")
        usage = data.get("usageMetadata") or {}
        return LLMReply(
            text=text, model=mid,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            eval_tokens=int(usage.get("candidatesTokenCount") or 0),
            seconds=time.time() - t0, truncated=False,
            thought_tokens=int(usage.get("thoughtsTokenCount") or 0))

    def chat_json(self, system: str, user: str, **kw):
        reply = self.chat(system, user, json_mode=True, **kw)
        try:
            return extract_json(reply.text), reply
        except ValueError:
            fixed = self.chat(
                "You convert text into valid JSON. Reply with JSON only.",
                f"Convert this into the valid JSON it was meant to be:\n\n"
                f"{reply.text[:4000]}", json_mode=True, temperature=0.0)
            return extract_json(fixed.text), reply

    # -- embeddings -------------------------------------------------------
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        mid = model or self.embed_model
        out = []
        key = self._require_key()
        with httpx.Client(timeout=settings.llm_timeout) as c:
            for t in texts:
                r = c.post(f"{BASE}/models/{mid}:embedContent",
                           headers={"x-goog-api-key": key},
                           json={"model": f"models/{mid}",
                                 "content": {"parts": [{"text": t}]}})
                if r.status_code >= 400:
                    raise LLMUnavailable(
                        f"Gemini embeddings failed ({r.status_code}).")
                out.append((r.json().get("embedding") or {}).get("values", []))
        return out

    def embed_batch(self, texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT",
                    dims: int | None = 768, model: str | None = None) -> list[list[float]]:
        """Many texts per request: 1,000 SKUs is 10 calls, not 1,000.

        task_type matters: documents and queries are embedded differently, so
        the catalogue uses RETRIEVAL_DOCUMENT and a look's item descriptions use
        RETRIEVAL_QUERY. dims truncates the vector (768 keeps quality and makes
        the index a quarter of the size); a model that refuses it gets its
        native size instead.
        """
        mid = model or self.embed_model
        out: list[list[float]] = []
        for i in range(0, len(texts), 100):            # API limit per batch
            reqs = []
            for t in texts[i:i + 100]:
                rq = {"model": f"models/{mid}", "content": {"parts": [{"text": t}]},
                      "taskType": task_type}
                if dims:
                    rq["outputDimensionality"] = dims
                reqs.append(rq)
            try:
                data = self._post(f"models/{mid}:batchEmbedContents",
                                  {"requests": reqs}, model=mid)
            except LLMUnavailable as e:
                if dims and "returned 400" in str(e):
                    return self.embed_batch(texts, task_type=task_type, dims=None,
                                            model=mid)
                raise
            out.extend(e.get("values", []) for e in data.get("embeddings", []))
        if len(out) != len(texts):
            raise LLMUnavailable(f"Gemini returned {len(out)} embeddings for "
                                 f"{len(texts)} texts.")
        return out
