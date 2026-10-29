"""Request/response schemas."""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class SizeProfile(BaseModel):
    unit: str = "cm"
    gender: str = "male"
    chest: float | None = None
    waist: float | None = None
    hips: float | None = None
    height: float | None = None
    foot: float | None = None
    shoulder: float | None = None
    frontLength: float | None = None
    neck: float | None = None
    thigh: float | None = None
    fitPref: str = "true"


class SearchRequest(BaseModel):
    q: str
    limit: int = 24
    useModel: bool = True
    profile: dict[str, Any] | None = None


class SyncRequest(BaseModel):
    products: list[dict[str, Any]]
    replace: bool = False


class FitRequest(BaseModel):
    sku: str
    size: str | None = None
    profile: SizeProfile
    product: dict[str, Any] | None = None


class AvatarRequest(BaseModel):
    profile: SizeProfile
    photoId: str | None = None


class TryOnRequest(BaseModel):
    profile: SizeProfile
    items: list[dict[str, Any]] = Field(default_factory=list)
    photoId: str | None = None


class LLMSettings(BaseModel):
    """Runtime model configuration. `gemini_api_key` is write-only: it is held
    in server memory and no endpoint ever reads it back."""
    provider: str | None = None            # ollama | gemini | auto
    gemini_api_key: str | None = None
    gemini_text_model: str | None = None
    gemini_vision_model: str | None = None
    ollama_host: str | None = None
    text_model: str | None = None
    vision_model: str | None = None
    num_ctx: int | None = None


class PaletteRequest(BaseModel):
    undertone: str = "neutral"       # warm | cool | neutral
    depth: str = "medium"            # light | medium | deep
    season: str | None = None        # override the derived season
    audience: str | None = None
    perRow: int = 8
    budgetMax: float | None = None
