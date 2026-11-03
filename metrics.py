"""Evaluation metrics: sketch preservation (edge based) and prompt fidelity (CLIP score)."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from config import CLIP_MODEL


def canny_edges(image: Image.Image, low: int = 100, high: int = 200) -> np.ndarray:
    """Boolean Canny edge map of an image."""
    gray = cv2.GaussianBlur(np.array(image.convert("L")), (5, 5), 0)
    return cv2.Canny(gray, low, high) > 0


def sketch_preservation(control: Image.Image, output: Image.Image, tol: int = 6) -> dict[str, float]:
    """Compare the control sketch (white lines) with the Canny edges of the generated image.

    recall     share of sketch pixels lying within `tol` px of an output edge (main metric)
    precision  share of output edge pixels within `tol` px of the sketch. It is naturally low
               because the model adds texture and detail edges that the sketch does not contain
    iou        intersection over union of the two tolerance-dilated edge maps
    chamfer    mean distance (px) from each sketch pixel to the nearest output edge (lower = better)
    """
    sketch = np.array(control.convert("L")) > 127
    edges = canny_edges(output.resize(control.size))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    sketch_dil = cv2.dilate(sketch.astype(np.uint8), kernel) > 0
    edges_dil = cv2.dilate(edges.astype(np.uint8), kernel) > 0
    dist_to_edges = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)

    return {
        "recall": float((sketch & edges_dil).sum() / max(sketch.sum(), 1)),
        "precision": float((edges & sketch_dil).sum() / max(edges.sum(), 1)),
        "iou": float((sketch_dil & edges_dil).sum() / max((sketch_dil | edges_dil).sum(), 1)),
        "chamfer": float(dist_to_edges[sketch].mean()) if sketch.any() else float("nan"),
    }


class ClipScorer:
    """CLIP score: 100 * cosine similarity between image and text embeddings."""

    def __init__(self, model_id: str = CLIP_MODEL) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def __call__(self, image: Image.Image, text: str) -> float:
        torch = self._torch
        inputs = self.processor(
            text=[text], images=image, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        similarity = torch.nn.functional.cosine_similarity(out.image_embeds, out.text_embeds)
        return 100.0 * similarity.item()
