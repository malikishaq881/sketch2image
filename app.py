"""Gradio demo: upload a sketch, describe it, and generate an image.

    python app.py            # local
    python app.py --share    # public link (useful on Google Colab)
"""
from __future__ import annotations

import argparse

import gradio as gr

from config import NEGATIVE_PROMPT, STYLES, GenerationSettings, build_prompt
from metrics import sketch_preservation
from pipeline import generate, load_pipeline, preprocess_sketch

_pipe = None


def _get_pipeline():
    """Load the model on first use."""
    global _pipe
    if _pipe is None:
        _pipe = load_pipeline()
    return _pipe


def run(sketch, subject, style, extra, cond_scale, guidance, steps, seed):
    if sketch is None:
        raise gr.Error("Please upload a sketch.")
    if not subject or not subject.strip():
        raise gr.Error("Please describe the subject, e.g. 'a futuristic skyscraper'.")

    control = preprocess_sketch(sketch)
    settings = GenerationSettings(steps=int(steps), guidance=guidance,
                                  cond_scale=cond_scale, seed=int(seed))
    image = generate(_get_pipeline(), control, build_prompt(subject, style, extra),
                     settings, NEGATIVE_PROMPT)
    m = sketch_preservation(control, image)
    report = f"Sketch recall: {m['recall']:.2f}  |  IoU: {m['iou']:.2f}  |  Chamfer: {m['chamfer']:.1f}px"
    return control, image, report


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Sketch to Image") as demo:
        gr.Markdown("# Sketch → Image\nStable Diffusion 1.5 + ControlNet Scribble")
        with gr.Row():
            with gr.Column():
                sketch = gr.Image(type="pil", label="Sketch (dark lines on light paper work best)")
                subject = gr.Textbox(label="Subject", placeholder="a futuristic skyscraper")
                style = gr.Dropdown(list(STYLES), value="photorealistic", label="Style")
                extra = gr.Textbox(label="Extra prompt (optional)", placeholder="sunset, glass and steel")
                cond = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="ControlNet conditioning scale (sketch strength)")
                guidance = gr.Slider(1, 20, 7.5, step=0.5, label="Guidance scale (prompt strength)")
                steps = gr.Slider(10, 60, 30, step=1, label="Steps")
                seed = gr.Number(42, label="Seed", precision=0)
                button = gr.Button("Generate", variant="primary")
            with gr.Column():
                control_out = gr.Image(label="Control map")
                result = gr.Image(label="Result")
                report = gr.Textbox(label="Sketch preservation")
        button.click(run, [sketch, subject, style, extra, cond, guidance, steps, seed],
                     [control_out, result, report])
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    build_demo().launch(share=parser.parse_args().share)
