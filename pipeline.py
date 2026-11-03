"""Sketch-to-image pipeline: Stable Diffusion 1.5 + ControlNet (scribble)."""
from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image, ImageOps

from config import (
    BASE_MODEL,
    CONTROLNET_MODEL,
    IMAGE_SIZE,
    NEGATIVE_PROMPT,
    GenerationSettings,
)

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Return 'cuda' when a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(
    base_model: str = BASE_MODEL,
    controlnet_model: str = CONTROLNET_MODEL,
    show_progress: bool = False,
) -> StableDiffusionControlNetPipeline:
    """Load Stable Diffusion 1.5 with a ControlNet branch (fp16 on GPU, fp32 on CPU)."""
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info("Loading %s + %s on %s (%s)", base_model, controlnet_model, device, dtype)

    controlnet = ControlNetModel.from_pretrained(controlnet_model, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=not show_progress)
    return pipe.to(device)


def preprocess_sketch(
    image: Image.Image,
    size: int = IMAGE_SIZE,
    thicken: int = 1,
    threshold: int = 100,
) -> Image.Image:
    """Convert a sketch into a ControlNet scribble map (white lines on black, size x size).

    Steps: autocontrast -> pad to a square (aspect ratio preserved) -> invert paper-style
    sketches (dark lines on light paper) -> binarise -> optionally thicken the strokes.
    """
    gray = ImageOps.autocontrast(image.convert("L"))
    light_background = float(np.array(gray).mean()) > 127
    fill = 255 if light_background else 0

    gray = ImageOps.contain(gray, (size, size), Image.LANCZOS)
    canvas = Image.new("L", (size, size), fill)
    canvas.paste(gray, ((size - gray.width) // 2, (size - gray.height) // 2))
    if light_background:
        canvas = ImageOps.invert(canvas)

    lines = (np.array(canvas) > threshold).astype(np.uint8) * 255
    if thicken > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thicken + 1, 2 * thicken + 1))
        lines = cv2.dilate(lines, kernel)
    return Image.fromarray(lines).convert("RGB")


def generate(
    pipe: StableDiffusionControlNetPipeline,
    control: Image.Image,
    prompt: str,
    settings: GenerationSettings = GenerationSettings(),
    negative_prompt: str = NEGATIVE_PROMPT,
    size: int = IMAGE_SIZE,
) -> Image.Image:
    """Generate one image from a preprocessed control map and a text prompt."""
    generator = torch.Generator(device="cpu").manual_seed(int(settings.seed))
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=control,
        num_inference_steps=int(settings.steps),
        guidance_scale=float(settings.guidance),
        controlnet_conditioning_scale=float(settings.cond_scale),
        generator=generator,
        height=size,
        width=size,
    )
    return result.images[0]
