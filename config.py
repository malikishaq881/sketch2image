"""Shared configuration: model identifiers, prompt styles and default generation settings."""
from __future__ import annotations

from dataclasses import dataclass

# --- Models (Hugging Face Hub) -------------------------------------------------------------
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_scribble"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# --- Generation ----------------------------------------------------------------------------
IMAGE_SIZE = 512

NEGATIVE_PROMPT = (
    "lowres, blurry, deformed, extra limbs, text, watermark, bad anatomy, cropped"
)

STYLES: dict[str, str] = {
    "photorealistic": "photorealistic, ultra detailed, 8k, natural lighting, sharp focus",
    "cyberpunk": "cyberpunk style, neon lights, night, rain reflections, cinematic",
    "watercolor": "watercolor painting, soft pastel colors, paper texture, artistic",
}


@dataclass(frozen=True)
class GenerationSettings:
    """Settings that control a single generation.

    steps       number of denoising steps
    guidance    classifier-free guidance scale (how strongly the text prompt is followed)
    cond_scale  ControlNet conditioning scale (how strongly the sketch is enforced)
    seed        random seed (fixes the initial noise)
    """

    steps: int = 30
    guidance: float = 7.5
    cond_scale: float = 1.0
    seed: int = 42


def build_prompt(subject: str, style: str | None = None, extra: str = "") -> str:
    """Combine a subject, an optional named style and optional free text into one prompt."""
    parts = [subject.strip(), STYLES[style] if style else "", extra.strip()]
    return ", ".join(p for p in parts if p)
