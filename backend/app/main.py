"""StyleMate API.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import base64
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import guard
from .catalogue.store import catalogue
from .config import DOTENV_LOADED, DOTENV_PATH, settings
from .llm.router import LLMUnavailable, llm
from .models import (AvatarRequest, FitRequest, LLMSettings, PaletteRequest,
                     SearchRequest, SizeProfile, SyncRequest, TryOnRequest)
from .services import avatar as avatar_svc
from .services import fit as fit_svc
from .services import size_labels
from .services import garment as garment_svc
from .services import look as look_svc
from .services import palette as palette_svc
from .services import search as search_svc
from .services import semantic as semantic_svc

app = FastAPI(title="StyleMate API", version="1.0.0")


@app.on_event("startup")
def _warm_semantic_index() -> None:
    # Embedding 1,000 SKUs takes a few seconds once (then it is cached on
    # disk); do it now rather than inside the first shopper's request.
    semantic_svc.warm_in_background()


# CORS lets OTHER websites call this API from their visitors' browsers. The page
# is served from this same server, so it never needs CORS. Locally it stays open
# (the standalone HTML file calls in from file://); hosted, it is off unless
# SM_CORS names the origins to allow.
if not guard.PUBLIC or os.getenv("SM_CORS"):
    app.add_middleware(
        CORSMiddleware, allow_origins=list(settings.cors_origins) or ["*"],
        allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD = 12 * 1024 * 1024          # 12 MB
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
STATIC = Path(__file__).parent / "static"
ASSETS = settings.cache_dir.parent / "garment_assets"
ASSETS.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- misc
@app.get("/api/ping")
def ping() -> dict:
    """For the host's health check: answers without calling any model, so a
    check every few seconds costs nothing."""
    return {"ok": True}


@app.get("/api/health")
def health() -> dict:
    h = llm.health()
    return {
        "ok": True,
        "catalogue": {"products": len(catalogue),
                      "lastSync": catalogue.get_meta("lastSync")},
        "llm": h,
        "guard": guard.status(),
        "features": {
            "search": {"ready": True,
                       "aiUnderstanding": h.get("textModelReady", False),
                       "note": "Rule parsing always works; the model adds "
                               "sentence understanding."},
            "shopALook": {"ready": True,
                          "visionModel": h.get("visionModelReady", False),
                          "semanticIndex": semantic_svc.index.status(),
                          "note": "Zone analysis always works; a vision model "
                                  "adds garment recognition."},
            "fittingRoom": {"ready": True, "clothSimulation": False,
                            "supplierMeshes": any(ASSETS.glob("*.gl*")),
                            "note": "Avatar and fit maths are local. Garment "
                                    "shells are measurement-derived, not cloth "
                                    "simulation."},
        },
    }


# A page prompt is a few thousand characters; the largest (the legacy
# full-catalogue pipeline) is ~60k. Anything far beyond that is not the page.
MAX_PROMPT_CHARS = 120_000


@app.post("/api/llm/chat")
def llm_chat(req: dict, _q: None = Depends(guard.ai_quota)) -> dict:
    """One model call on behalf of the storefront page.

    The page used to call Ollama straight from the browser. That cannot work for
    Gemini: an API key in a web page is readable by anyone who opens the page, so
    the key stays here and the page asks the backend instead. Which provider
    answers is decided by SM_PROVIDER and whether a key is set — the page does
    not know and does not need to.

    SECURITY: this is a relay to a paid API. On localhost that is fine. Before
    this backend is exposed to the internet it needs authentication and a rate
    limit, or anyone who finds the URL can spend the key's quota.
    """
    msgs = req.get("messages") or []
    if not isinstance(msgs, list) or not msgs:
        raise HTTPException(400, "messages[] is required.")

    # Both clients take one system string and one user string. The page may send
    # a repair turn (system, user, assistant, user), so the extra turns are
    # folded into the user message with their role named.
    system_parts: list[str] = []
    user_parts: list[str] = []
    images: list[bytes] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user").lower()
        content = str(m.get("content") or "")
        for b64 in (m.get("images") or [])[:4]:
            # validate=True matters: without it b64decode silently discards any
            # character outside the base64 alphabet, so "!!!" decodes to b"" and
            # a corrupt upload reaches the model as an empty image.
            try:
                raw = base64.b64decode(str(b64), validate=True)
            except Exception:
                raise HTTPException(400, "images[] entries must be base64.")
            if not raw:
                raise HTTPException(400, "images[] contains an empty image.")
            images.append(raw)
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            user_parts.append("[your previous reply]\n" + content)
        elif content:
            user_parts.append(content)

    if sum(len(b) for b in images) > MAX_UPLOAD:
        raise HTTPException(413, "Images too large.")
    if sum(len(p) for p in system_parts + user_parts) > MAX_PROMPT_CHARS:
        raise HTTPException(413, "Prompt too long.")

    t0 = time.time()
    try:
        reply = llm.chat(
            "\n\n".join(p for p in system_parts if p).strip(),
            "\n\n".join(p for p in user_parts if p).strip(),
            images=images or None,
            json_mode=bool(req.get("json", True)),
            # Hosted, the server's configured model is the only one: a visitor
            # cannot ask the relay for a pricier model on the owner's key.
            model=(None if guard.PUBLIC else (req.get("model") or None)),
        )
    except LLMUnavailable as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:                       # noqa: BLE001 - surfaced to the page
        raise HTTPException(502, f"{type(exc).__name__}: {exc}")

    return {
        "text": reply.text,
        "model": reply.model,
        "provider": llm.active_name,
        "promptTokens": getattr(reply, "prompt_tokens", 0),
        "evalTokens": getattr(reply, "eval_tokens", 0),
        "ms": int((time.time() - t0) * 1000),
    }


@app.get("/api/config")
def frontend_config() -> dict:
    """Non-secret settings the browser page adopts at boot.

    The page cannot read backend/.env, so the server publishes the half of it
    that is safe to publish. GEMINI_API_KEY is deliberately NOT here and must
    never be added: anything returned by this endpoint is readable by anyone
    who can open the page. Only whether a key is present is reported.
    """
    return {
        "ollamaHost": settings.ollama_host,
        "textModel": settings.text_model,
        "visionModel": settings.vision_model,
        "numCtx": settings.num_ctx,
        "llmTimeout": settings.llm_timeout,
        "results": settings.results_size,
        "provider": settings.provider,
        "geminiKeyPresent": bool(settings.gemini_api_key),
        # Hosted, the audit trail goes to this server (and from there to the
        # host's log viewer). localhost:8765 would be each VISITOR's machine.
        "logSink": "/api/audit/log" if guard.PUBLIC else settings.log_sink,
        "public": guard.PUBLIC,
        "dotenv": {"path": str(DOTENV_PATH), "loaded": DOTENV_LOADED},
    }


@app.get("/api/settings/llm")
def llm_settings() -> dict:
    """Current provider state. Never returns the API key itself."""
    return llm.health_all()


@app.post("/api/settings/llm")
def set_llm_settings(cfg: LLMSettings) -> dict:
    # Hosted, one visitor must not be able to swap the key, provider or model
    # for everyone. The owner changes them in the host's environment variables.
    guard.settings_locked()
    return llm.configure(**cfg.model_dump(exclude_none=True))


@app.get("/api/settings/gemini/models")
def gemini_models() -> dict:
    """Live model list for the key that is set, or the known-good fallback."""
    return llm.gemini.list_models()


@app.get("/api/catalogue/stats")
def cat_stats() -> dict:
    return {"count": len(catalogue), "facets": catalogue.facets(),
            "lastSync": catalogue.get_meta("lastSync"),
            "pinned": catalogue.pinned}


@app.post("/api/catalogue/sync")
def cat_sync(req: SyncRequest) -> dict:
    # A pinned catalogue file is the source of truth. The page still pushes
    # what it loaded (stylemate-room.js does it before every search); ignoring
    # it keeps a stale tab from putting old rows back on the shelf.
    if catalogue.pinned:
        return {"stored": 0, "total": len(catalogue), "pinned": catalogue.pinned}
    if not req.products:
        raise HTTPException(400, "No products supplied.")
    n = catalogue.upsert(req.products)
    catalogue.set_meta("lastSync", time.strftime("%Y-%m-%d %H:%M:%S"))
    return {"stored": n, "total": len(catalogue)}


# ------------------------------------------------------------------- search
@app.post("/api/search")
def api_search(req: SearchRequest, _q: None = Depends(guard.ai_quota)) -> dict:
    if not req.q.strip():
        raise HTTPException(400, "Empty query.")
    return search_svc.search(req.q, limit=req.limit, use_model=req.useModel,
                             profile=req.profile)


# ------------------------------------------------------------------ palette
@app.post("/api/palette")
def api_palette(req: PaletteRequest) -> dict:
    """Seasonal palette for a skin reading, plus products in those colours."""
    return palette_svc.recommend(
        req.undertone, req.depth, season_key=req.season,
        audience=req.audience, per_row=max(1, min(req.perRow, 24)),
        budget_max=req.budgetMax)


@app.get("/api/palette/seasons")
def api_seasons() -> dict:
    return {"seasons": [s.dict() for s in palette_svc.SEASONS.values()]}


# -------------------------------------------------------------- shop a look
def _read_upload(f: UploadFile) -> bytes:
    if f.content_type not in ALLOWED:
        raise HTTPException(415, f"Only JPEG, PNG or WebP. Got {f.content_type}.")
    data = f.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image larger than 12 MB.")
    if not data:
        raise HTTPException(400, "Empty file.")
    return data


@app.post("/api/look")
def api_look(photo: UploadFile = File(...), audience: str = Form(""),
             perItem: int = Form(6), useVision: int = Form(1),
             _q: None = Depends(guard.ai_quota)) -> dict:
    data = _read_upload(photo)
    return look_svc.analyse(data, audience=(audience or None),
                            per_item=max(1, min(perItem, 12)),
                            use_vision=bool(useVision))


@app.post("/api/look/match")
def api_look_match(req: dict, _q: None = Depends(guard.ai_quota)) -> dict:
    """Re-run the catalogue match from a look's text description.

    Takes the `items` that /api/look returned (each item's `detected` block)
    and matches them again - for another audience, or more results. No image
    is sent and no vision call is made."""
    items = req.get("items") or []
    if not isinstance(items, list) or not items:
        raise HTTPException(400, "items[] is required.")
    aud = str(req.get("audience") or "") or None
    per = max(1, min(int(req.get("perItem") or 6), 12))
    return look_svc.rematch([it.get("detected", it) if isinstance(it, dict) else {}
                             for it in items], audience=aud, per_item=per)


# ---------------------------------------------------------------------- fit
def _product_for(sku: str, supplied: dict | None) -> dict:
    p = catalogue.get(sku)
    if p:
        try:
            raw = json.loads(p.get("raw") or "{}")
            return {**raw, **{k: v for k, v in p.items() if v}}
        except json.JSONDecodeError:
            return p
    if supplied:
        return supplied
    raise HTTPException(404, f"Unknown SKU {sku}. Sync the catalogue first.")


@app.post("/api/fit")
def api_fit(req: FitRequest) -> dict:
    product = _product_for(req.sku, req.product)
    body = req.profile.model_dump()
    unit = body.pop("unit", "cm")
    return _fit_in_labels(product, req.size, body, unit)


def _fit_in_labels(product: dict, label: str | None, body: dict, unit: str) -> dict:
    """Fit report for one of the product's own size labels (or its best one)."""
    chart = size_labels.to_chart(label, product) if label else None
    if not chart:
        chart = fit_svc.best_size(product, body, unit) or "M"
        label = None if not label or not size_labels.to_chart(label, product) else label
    report = fit_svc.evaluate(product, chart, body, unit).dict()
    return size_labels.report_in_labels(report, label, product)


@app.post("/api/fit/batch")
def api_fit_batch(req: TryOnRequest) -> dict:
    body = req.profile.model_dump()
    unit = body.pop("unit", "cm")
    out = []
    for it in req.items:
        sku = str(it.get("sku") or it.get("id") or "")
        if not sku:
            continue
        product = _product_for(sku, it.get("product"))
        out.append(_fit_in_labels(product, it.get("size"), body, unit))
    return {"reports": out}


# ------------------------------------------------------------------- avatar
PHOTOS: dict[str, dict] = {}        # photoId -> {bytes, evidence}
PHOTOS_MAX = 64                     # oldest dropped first: memory must not grow with traffic


@app.post("/api/avatar/photo")
def api_avatar_photo(photo: UploadFile = File(...)) -> dict:
    """Analyse a body photo. The image is held in memory for this process
    only, is never written to a public path, and is dropped by /api/avatar/forget."""
    data = _read_upload(photo)
    ev = avatar_svc.analyse_photo(data)
    pid = secrets.token_urlsafe(12)
    PHOTOS[pid] = {"bytes": data, "evidence": ev}
    while len(PHOTOS) > PHOTOS_MAX:
        PHOTOS.pop(next(iter(PHOTOS)))
    return {"photoId": pid, "evidence": ev.dict(),
            "note": "The photo stays in server memory and is not served from "
                    "any URL. Use Forget my photo to remove it."}


@app.post("/api/avatar/forget")
def api_avatar_forget(photoId: str = Form(...)) -> dict:
    existed = PHOTOS.pop(photoId, None) is not None
    return {"removed": existed}


@app.post("/api/avatar")
def api_avatar(req: AvatarRequest) -> dict:
    prof = req.profile.model_dump()
    m, supplied, defaulted = avatar_svc.resolve_measurements(prof)
    ev = None
    notes: list[str] = []
    if req.photoId:
        rec = PHOTOS.get(req.photoId)
        if not rec:
            notes.append("That photo is no longer held on the server; "
                         "the avatar was built from measurements alone.")
        else:
            ev = rec["evidence"]
            m, ev_notes = avatar_svc.apply_evidence(m, ev)
            notes.extend(ev_notes)
            notes.extend(ev.notes)
    mesh = avatar_svc.build_mesh(m, evidence=ev)
    clean = {k: v for k, v in m.items() if not k.startswith("_")}
    return {
        "mesh": mesh,
        "measurementsCm": clean,
        "supplied": supplied,
        "defaulted": defaulted,
        "skinRgb": list(ev.skin_rgb) if (ev and ev.skin_rgb) else None,
        "photoEvidence": ev.dict() if ev else None,
        "notes": notes,
        "method": "parametric-anthropometric",
        "disclosure": (
            "This is a parametric mesh built from your measurements, not a 3D "
            "scan. A single photograph carries no depth, so it is used only to "
            "check proportions and sample skin tone - never to infer your "
            "body shape. Measurements you have not entered use a population "
            "default and are listed under 'defaulted'."),
    }


@app.post("/api/fittingroom")
def api_fittingroom(req: TryOnRequest) -> dict:
    """Everything the 3D fitting room needs in one call."""
    prof = req.profile.model_dump()
    unit = prof.get("unit", "cm")
    m, supplied, defaulted = avatar_svc.resolve_measurements(prof)
    ev = None
    notes: list[str] = []
    if req.photoId and req.photoId in PHOTOS:
        ev = PHOTOS[req.photoId]["evidence"]
        m, ev_notes = avatar_svc.apply_evidence(m, ev)
        notes.extend(ev_notes + ev.notes)

    mesh = avatar_svc.build_mesh(m, evidence=ev)
    body = {k: v for k, v in prof.items() if k != "unit"}
    height_cm = m["height"]

    garments, reports = [], []
    for it in req.items:
        sku = str(it.get("sku") or it.get("id") or "")
        if not sku:
            continue
        product = _product_for(sku, it.get("product"))
        rep = _fit_in_labels(product, it.get("size"), body, unit)
        reports.append(rep)
        size = rep["chart_size"]
        shell = garment_svc.build_shell(product, size, m, height_cm)
        shell["name"] = product.get("name")
        shell["image"] = product.get("image")
        shell["colour"] = product.get("colour")
        shell["overall"] = report.overall
        garments.append(shell)

    return {
        "mesh": mesh,
        "measurementsCm": {k: v for k, v in m.items() if not k.startswith("_")},
        "supplied": supplied, "defaulted": defaulted,
        "skinRgb": list(ev.skin_rgb) if (ev and ev.skin_rgb) else None,
        "garments": garments,
        "fitReports": reports,
        "notes": notes,
        "disclosure": (
            "The body is a parametric mesh from your measurements. Garment "
            "shapes are lofted from each item's size chart - they show how "
            "much room the garment has and where it ends, not the product's "
            "real cut. No cloth simulation is running."),
    }


# ------------------------------------------------------------------- assets
@app.get("/assets/garments/{name}")
def garment_asset(name: str):
    p = (ASSETS / name).resolve()
    if not str(p).startswith(str(ASSETS.resolve())) or not p.exists():
        raise HTTPException(404, "No such asset.")
    return FileResponse(p)


# ----------------------------------------------------------------- audit log
@app.get("/api/audit/ping")
def audit_ping() -> dict:
    return {"ok": True}


@app.post("/api/audit/log")
async def audit_log(request: Request, _q: None = Depends(guard.audit_quota)) -> dict:
    """The page's audit records, when hosted. One line per model call goes to
    stdout, which is the host's log viewer (Render: the service's Logs tab).
    Nothing is written to disk: a hosted disk is wiped on every deploy, and
    letting the public write files is a way to fill it."""
    body = await request.body()
    if len(body) > 400_000:
        raise HTTPException(413, "Log record too large.")
    try:
        d = json.loads(body.decode("utf-8"))
        rec = d.get("record") or {}
    except Exception:                                   # noqa: BLE001
        raise HTTPException(400, "Not a log record.")

    def f(k):
        v = rec.get(k)
        return "-" if v in (None, "") else str(v)[:60]
    print(f"[audit] {f('label')} {'update' if d.get('rewrite') else 'call'} "
          f"status={f('status')} model={f('served_by_model')} "
          f"prompt_tokens={f('prompt_tokens')} output_tokens={f('completion_tokens')} "
          f"latency_ms={f('latency_ms')} shown={f('products_shown')} "
          f"error={f('error_kind')}", flush=True)
    return {"ok": True}


_CAT_JS: dict = {}


@app.get("/catalogue.js")
def catalogue_js() -> Response:
    """The page loads <script src="catalogue.js"> before anything else. Serving
    it from the pinned file means the page and the backend can never disagree
    about what is on the shelf, and there is no copy to keep in step."""
    pin = catalogue.pinned
    if not pin:
        return Response("/* no pinned catalogue */", media_type="application/javascript")
    if _CAT_JS.get("hash") != pin["hash"]:
        body = Path(pin["path"]).read_text(encoding="utf-8")
        _CAT_JS.update(hash=pin["hash"], js="window.STYLEMATE_CATALOGUE=" + body + ";")
    return Response(_CAT_JS["js"], media_type="application/javascript",
                    headers={"Cache-Control": "no-cache"})


_SLIM: dict = {}
_SLIM_DROP = ("reviews", "details", "desc", "doc", "raw", "url", "sourceSku",
              "masterCategory", "subCategory", "year", "spec", "size", "image")


def _full(r: dict) -> dict:
    """A store row is the indexed columns plus the original product JSON in `raw`."""
    try:
        raw = json.loads(r.get("raw") or "{}")
    except ValueError:
        raw = {}
    return {**raw, **{k: v for k, v in r.items() if k not in ("raw", "doc") and v not in ("", None)}}


@app.get("/api/catalogue")
def catalogue_slim() -> Response:
    """Card-sized rows for the v16 page; reviews and details come per product."""
    pin = catalogue.pinned or {}
    if _SLIM.get("hash") != pin.get("hash"):
        rows = [{k: v for k, v in _full(r).items() if k not in _SLIM_DROP} for r in catalogue.rows]
        _SLIM.update(hash=pin.get("hash"), body=json.dumps(
            {"source": pin.get("source"), "total": len(rows), "rows": rows},
            ensure_ascii=False, separators=(",", ":")))
    return Response(_SLIM["body"], media_type="application/json",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/product/{pid}")
def product_full(pid: str) -> dict:
    p = catalogue.get(pid)
    if not p:
        raise HTTPException(404, "No such product.")
    return _full(p)


IMG_DIR = Path(__file__).parent / "data" / "img"


@app.get("/img/{name}")
def product_image(name: str) -> FileResponse:
    path = (IMG_DIR / Path(name).name)
    if path.suffix != ".jpg" or not path.is_file():
        raise HTTPException(404, "No such image.")
    return FileResponse(path, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


if STATIC.exists():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
