# Sketch-to-Image with Stable Diffusion + ControlNet

**Author:** Malik Muhammad Ishaq – International Master of Biometrics and Intelligent Vision, Université Paris-Est Créteil (UPEC)

Turns a rough hand-drawn sketch into a detailed image while keeping the sketch's structure.
Project for *Generative AI – Bringing Sketches to Life*.

## Method
- **Base model:** Stable Diffusion 1.5 (latent diffusion; a U-Net denoises in the VAE latent space, guided by CLIP text embeddings).
- **Sketch control:** ControlNet Scribble (`lllyasviel/control_v11p_sd15_scribble`). ControlNet is a trainable copy of the U-Net encoder, connected to the frozen SD U-Net through zero-initialised convolutions. The preprocessed sketch (white lines on black) is fed to this branch and its features are added to the U-Net at every denoising step, so the layout of the image follows the strokes while the text prompt decides content, style, material and lighting.
- **Preprocessing (`pipeline.py`):** autocontrast → pad to square → invert (paper sketches) → binarise → thicken strokes.
- **Key settings:** `controlnet_conditioning_scale` (sketch strength), `guidance_scale` (prompt strength), number of steps, seed, prompt style.
- **Metrics (`metrics.py`):**
  - *Sketch preservation:* Canny edges of the output vs the control sketch, with a pixel tolerance → recall, precision, IoU, Chamfer distance. Recall and Chamfer are the main indicators (precision is naturally low because the model adds texture edges).
  - *Prompt fidelity:* CLIP score (ViT-B/32).

## Installation
```bash
git clone https://github.com/malikishaq881/sketch2image.git && cd sketch2image
pip install -r requirements.txt
```
A GPU is strongly recommended (tested target: Google Colab T4, fp16, ~5 s per 512×512 image at 30 steps).
Or open `Sketch2Image_Colab.ipynb` in Google Colab (Runtime → T4 GPU).

## Usage
```bash
python make_sample_sketches.py                    # optional placeholder sketches; replace with your own
python run_experiments.py --sketches sketches --out outputs --prompts prompts.json
python app.py --share                             # Gradio demo
```
Add `--quick` to `run_experiments.py` for a fast smoke test.
Name each sketch file after its subject, or map it in `prompts.json`.

Outputs per sketch (`outputs/<sketch>/`): `control.png`, every generated image, and comparison grids
`grid_style.png`, `grid_cond_scale.png`, `grid_guidance.png`, `grid_steps.png`, `grid_seed.png`.
Global: `outputs/results.csv` (all settings + metrics) and `outputs/plot_cond_scale.png`.

## Experiments
| Experiment | Values | Fixed |
|---|---|---|
| Prompt style | photorealistic, cyberpunk, watercolor | steps 30, guidance 7.5, scale 1.0, seed 42 |
| Conditioning scale | 0.3, 0.6, 1.0, 1.3, 1.6 | rest default |
| Guidance scale | 5, 7.5, 12 | rest default |
| Steps | 15, 30, 50 | rest default |
| Seed | 1, 2, 3, 4 | rest default |

## References
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022.
- Zhang, Rao, Agrawala, *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet), ICCV 2023.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021.
- Models: `stable-diffusion-v1-5/stable-diffusion-v1-5`, `lllyasviel/control_v11p_sd15_scribble`, `openai/clip-vit-base-patch32` (Hugging Face).
- Libraries: PyTorch, Hugging Face `diffusers`, `transformers`, `accelerate`, OpenCV, Gradio, pandas, matplotlib.
- Sketches: generated programmatically (hand-drawn style) with `make_sample_sketches.py` and the author's sketch script; replace with own hand-drawn sketches if used.

## AI assistance
The code was written with the help of Claude (Anthropic) and reviewed/understood by the author.
