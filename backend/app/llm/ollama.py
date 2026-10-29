"""Thin Ollama client.

Three things this handles that a bare fetch does not:
  * JSON mode with a repair pass - small models fence their JSON or trail commas.
  * num_ctx on the native /api/chat route, with the real prompt token count
    echoed back so we can see when the catalogue is being truncated.
  * A single place that reports WHY the model is unreachable, so every feature
    can degrade to its deterministic path instead of erroring at the user.
"""
from __future__ import annotations

import base64, json, re, time
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import settings


class LLMUnavailable(RuntimeError):
    """Raised when the model cannot be reached or did not answer usefully."""


@dataclass
class LLMReply:
    text: str
    model: str
    prompt_tokens: int = 0
    eval_tokens: int = 0
    seconds: float = 0.0
    truncated: bool = False          # prompt exceeded the context window
    thought_tokens: int = 0          # Gemini bills reasoning as output tokens

    def as_meta(self) -> dict:
        return {
            "model": self.model,
            "promptTokens": self.prompt_tokens,
            "evalTokens": self.eval_tokens,
            "seconds": round(self.seconds, 2),
            "contextWindow": settings.num_ctx,
            "truncated": self.truncated,
        }


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> Any:
    """Pull the first JSON value out of a model reply."""
    if not text:
        raise ValueError("empty reply")
    m = _JSON_BLOCK.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Scan for the first balanced { } or [ ] - models like to prepend prose.
    # Try whichever bracket appears first so a top-level list is not mistaken
    # for its first object.
    cands = [(text.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if text.find(o) >= 0]
    for start, opener, closer in sorted(cands):
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    blob = text[start:i + 1]
                    blob = re.sub(r",\s*([}\]])", r"\1", blob)   # trailing commas
                    return json.loads(blob)
    raise ValueError("no JSON found in reply")


class Ollama:
    def __init__(self, host: str | None = None):
        self.host = (host or settings.ollama_host).rstrip("/")

    # -- plumbing ---------------------------------------------------------
    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        url = f"{self.host}{path}"
        try:
            with httpx.Client(timeout=timeout or settings.llm_timeout) as c:
                r = c.post(url, json=payload)
                r.raise_for_status()
                return r.json()
        except httpx.ConnectError as e:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve`."
            ) from e
        except httpx.TimeoutException as e:
            raise LLMUnavailable(
                f"{payload.get('model')} did not answer within "
                f"{timeout or settings.llm_timeout:.0f}s. A 7B model on CPU can take "
                "a while on the first call while weights page in."
            ) from e
        except httpx.HTTPStatusError as e:
            detail = e.response.text[:300]
            if e.response.status_code == 404:
                raise LLMUnavailable(
                    f"Model '{payload.get('model')}' is not pulled. "
                    f"Run `ollama pull {payload.get('model')}`."
                ) from e
            raise LLMUnavailable(f"Ollama returned {e.response.status_code}: {detail}") from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"Could not reach Ollama at {self.host}: {e}") from e

    # -- health -----------------------------------------------------------
    def health(self) -> dict:
        try:
            with httpx.Client(timeout=5) as c:
                r = c.get(f"{self.host}/api/tags")
                r.raise_for_status()
                tags = [m["name"] for m in r.json().get("models", [])]
        except Exception as e:                       # noqa: BLE001 - reported, not raised
            return {"reachable": False, "models": [], "error": str(e), "host": self.host}
        return {
            "reachable": True,
            "host": self.host,
            "models": tags,
            "textModel": settings.text_model,
            "textModelReady": _has(tags, settings.text_model),
            "visionModel": settings.vision_model,
            "visionModelReady": _has(tags, settings.vision_model),
            "embedModel": settings.embed_model,
            "embedModelReady": _has(tags, settings.embed_model),
        }

    # -- chat -------------------------------------------------------------
    def chat(self, system: str, user: str, *, images: list[bytes] | None = None,
             model: str | None = None, json_mode: bool = False,
             temperature: float = 0.2, timeout: float | None = None) -> LLMReply:
        if not settings.llm_enabled:
            raise LLMUnavailable("The model layer is switched off (SM_LLM_ENABLED=0).")
        msg: dict = {"role": "user", "content": user}
        if images:
            msg["images"] = [base64.b64encode(b).decode() for b in images]
        payload = {
            "model": model or settings.text_model,
            "messages": [{"role": "system", "content": system}, msg],
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": settings.num_ctx},
        }
        if json_mode:
            payload["format"] = "json"
        t0 = time.time()
        data = self._post("/api/chat", payload, timeout=timeout)
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        return LLMReply(
            text=(data.get("message") or {}).get("content", ""),
            model=payload["model"],
            prompt_tokens=prompt_tokens,
            eval_tokens=int(data.get("eval_count") or 0),
            seconds=time.time() - t0,
            truncated=prompt_tokens >= settings.num_ctx - 32,
        )

    def chat_json(self, system: str, user: str, **kw) -> tuple[Any, LLMReply]:
        reply = self.chat(system, user, json_mode=True, **kw)
        try:
            return extract_json(reply.text), reply
        except ValueError:
            # One repair attempt: hand the mess back and ask for JSON only.
            fixed = self.chat(
                "You convert text into valid JSON. Reply with JSON only, nothing else.",
                f"Convert this into the valid JSON it was meant to be:\n\n{reply.text[:4000]}",
                json_mode=True, temperature=0.0,
            )
            return extract_json(fixed.text), reply

    # -- embeddings -------------------------------------------------------
    def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        out = []
        m = model or settings.embed_model
        with httpx.Client(timeout=settings.llm_timeout) as c:
            for t in texts:
                try:
                    r = c.post(f"{self.host}/api/embeddings", json={"model": m, "prompt": t})
                    r.raise_for_status()
                    out.append(r.json()["embedding"])
                except Exception as e:                # noqa: BLE001
                    raise LLMUnavailable(f"Embedding model '{m}' unavailable: {e}") from e
        return out


def _has(tags: list[str], want: str) -> bool:
    """Ollama reports 'qwen2.5:7b'; a user may configure 'qwen2.5'."""
    want = want.strip()
    return any(t == want or t.split(":")[0] == want.split(":")[0] for t in tags)


llm = Ollama()
