"""Guards for running StyleMate on the public internet.

On localhost the backend has one trusted user. Hosted (SM_PUBLIC=1, which
render.yaml sets) anyone can reach every endpoint, and four of them spend the
owner's Gemini key. Visitors still never see the key - it stays on the server -
but without limits they could spend it. This bounds that:

  * a per-visitor rate limit on every AI endpoint   SM_AI_PER_MINUTE (15)
  * a ceiling on AI calls for the whole site per day SM_AI_PER_DAY   (600)
  * model settings are read-only: no visitor can swap the key, the provider
    or the model for everyone else

All of it is in memory, so it resets when the service restarts. That is the
right size for a demo on one instance; a busier deployment would move the
counters to Redis. Off entirely when SM_PUBLIC is not set.
"""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

PUBLIC = os.getenv("SM_PUBLIC", "").strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


PER_MINUTE = _int("SM_AI_PER_MINUTE", 15)
PER_DAY = _int("SM_AI_PER_DAY", 600)
AUDIT_PER_MINUTE = 60

_lock = threading.Lock()
_hits: dict[str, deque] = defaultdict(deque)
_day = {"date": "", "count": 0}


def client_ip(request: Request) -> str:
    # Render (like most hosts) sits behind a proxy: the visitor is the first
    # address in X-Forwarded-For, not the socket's peer.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _window(key: str, limit: int, now: float) -> int | None:
    """Sliding one-minute window. Returns seconds to wait, or None if allowed."""
    q = _hits[key]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        return int(60 - (now - q[0])) + 1
    q.append(now)
    return None


def ai_quota(request: Request) -> None:
    """FastAPI dependency for every endpoint that calls a paid model."""
    if not PUBLIC:
        return
    now = time.time()
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    with _lock:
        if _day["date"] != today:
            _day.update(date=today, count=0)
        if _day["count"] >= PER_DAY:
            raise HTTPException(
                429, "StyleMate's AI has reached today's limit on this demo site. "
                     "Search and browsing still work; AI features come back "
                     "tomorrow (UTC).")
        wait = _window("ai:" + client_ip(request), PER_MINUTE, now)
        if wait is not None:
            raise HTTPException(
                429, f"Too many AI requests from you in a minute. Try again in "
                     f"{wait} seconds.", headers={"Retry-After": str(wait)})
        _day["count"] += 1
        # Forget visitors who have gone quiet, so the table cannot grow forever.
        if len(_hits) > 5000:
            for k in [k for k, q in _hits.items() if not q or now - q[-1] > 60]:
                del _hits[k]


def audit_quota(request: Request) -> None:
    if not PUBLIC:
        return
    with _lock:
        if _window("audit:" + client_ip(request), AUDIT_PER_MINUTE, time.time()) is not None:
            raise HTTPException(429, "Too many log records.")


def settings_locked() -> None:
    if PUBLIC:
        raise HTTPException(
            403, "Model settings are locked on the hosted site. The owner changes "
                 "them in the hosting dashboard's environment variables.")


def status() -> dict:
    return {"public": PUBLIC, "aiPerMinute": PER_MINUTE, "aiPerDay": PER_DAY,
            "aiCallsToday": _day["count"] if _day["date"] else 0}
