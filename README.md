# Sketch-to-Image with Stable Diffusion and ControlNet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/malikishaq881/sketch2image/blob/main/Sketch2Image_Colab.ipynb)

Turns a rough line sketch into a detailed image while preserving the structure of the drawing.
Individual project for *Generative AI – Bringing Sketches to Life*.

**Author:** Malik Muhammad Ishaq, International Master of Biometrics and Intelligent Vision, Université Paris-Est Créteil (UPEC)

![Overview: sketch and three styles](results/overview.png)

## Contents
- [Method](#method)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Experiments and metrics](#experiments-and-metrics)
- [Results](#results)
- [Limitations](#limitations)
- [References](#references)

## Method

**Models**
- **Stable Diffusion 1.5**, a latent diffusion model. A U-Net removes noise step by step in the latent space of a VAE and is guided by CLIP text embeddings of the prompt.
- **ControlNet Scribble (v1.1)**, an additional network that adds spatial control. It is a trainable copy of the U-Net encoder attached to the frozen Stable Diffusion U-Net through zero-initialised convolutions.

**How the sketch influences the result.** The sketch is converted into a control map (white lines on black). ControlNet encodes this map and adds its features to the U-Net at every denoising step. The strokes therefore decide the layout, proportions and outlines, while the text prompt decides content, style, material and lighting. Two parameters balance them: `controlnet_conditioning_scale` (sketch strength) and `guidance_scale` (prompt strength).

**Pipeline**

```
sketch ──► preprocessing ──► control map ──┐
                                           ├─► SD 1.5 + ControlNet ──► image ──► metrics
text prompt (subject + style) ─────────────┘
```

Preprocessing (`pipeline.py`): autocontrast, pad to a square, invert paper-style sketches, binarise, thicken strokes.

## Repository structure

```
sketch2image/
├── Sketch2Image_Colab.ipynb   # run everything on Google Colab
├── config.py                  # models, styles, default settings
├── pipeline.py                # model loading, preprocessing, generation
├── metrics.py                 # sketch preservation and CLIP score
├── run_experiments.py         # experiments, grids, tables, plots
├── app.py                     # Gradio web demo
├── make_sketches.py           # creates the example sketches
├── prompts.json               # sketch name -> subject description
├── requirements.txt
├── sketches/                  # input sketches (building, robot, car, rocket)
└── results/                   # generated images, grids, metrics
```

## Installation

**Google Colab (recommended).** Open the notebook with the badge above, select *Runtime → Change runtime type → T4 GPU* and run all cells.

**Local machine.** Python 3.10+ and an NVIDIA GPU with at least 6 GB of VRAM are recommended. Install PyTorch for your CUDA version from <https://pytorch.org>, then:

```bash
git clone https://github.com/malikishaq881/sketch2image.git
cd sketch2image
pip install -r requirements.txt
```

The models (about 5 GB) are downloaded automatically from Hugging Face on the first run.

## Usage

```bash
# all experiments on every sketch in sketches/
python run_experiments.py --sketches sketches --prompts prompts.json --out results

# fast smoke test
python run_experiments.py --quick

# web demo (add --share for a public link)
python app.py
```

To use your own sketches, place PNG/JPG files (dark lines on a light background) in `sketches/` and add one entry per file to `prompts.json`, for example `"car": "a futuristic sports car"`.

## Reproducibility

| Item | Value |
|---|---|
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| ControlNet | `lllyasviel/control_v11p_sd15_scribble` |
| CLIP (metric) | `openai/clip-vit-base-patch32` |
| Scheduler | UniPC multistep |
| Resolution | 512 × 512, fp16 on GPU |
| Default settings | 30 steps, guidance 7.5, conditioning scale 1.0, seed 42 |
| Negative prompt | `lowres, blurry, deformed, extra limbs, text, watermark, bad anatomy, cropped` |
| Prompt template | `<subject>, <style description>` (styles in `config.py`) |

Every run writes `results/run_info.json` with the exact library versions, models, settings and arguments. Seeds are fixed, but results can differ slightly between GPU types and library versions.

## Experiments and metrics

| Experiment | Values | Other settings |
|---|---|---|
| Prompt style | photorealistic, cyberpunk, watercolor | defaults |
| Conditioning scale | 0.3, 0.6, 1.0, 1.3, 1.6 | defaults, photorealistic |
| Guidance scale | 5, 7.5, 12 | defaults, photorealistic |
| Steps | 15, 30, 50 | defaults, photorealistic |
| Seed | 1, 2, 3, 4 | defaults, photorealistic |

**Sketch preservation.** Canny edges of the generated image are compared with the control sketch using a 6-pixel tolerance. *Recall* (share of sketch pixels close to an output edge) and *Chamfer distance* (mean distance from sketch pixels to the nearest output edge) are the main indicators; *IoU* is also reported. *Precision* is naturally low because the model adds texture edges that are not in the sketch.

**Prompt fidelity.** CLIP score, 100 × cosine similarity between image and prompt embeddings.

Outputs: `results/<sketch>/` (control map, all images, one grid per experiment), `results/results.csv`, `results/summary.csv`, `results/overview.png`, `results/plot_cond_scale.png`.

## Results

![Effect of the conditioning scale](results/plot_cond_scale.png)

Per-experiment grids are in `results/<sketch>/grid_*.png`; all numbers are in `results/summary.csv`.

## Limitations
- Stable Diffusion 1.5 can misinterpret very abstract or incomplete sketches.
- Edge-based preservation metrics depend on the Canny thresholds and do not measure semantic correctness.
- The four example sketches are simple line drawings; complex scenes may need a higher conditioning scale or a more detailed prompt.

## References
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022. <https://arxiv.org/abs/2112.10752>
- Zhang, Rao, Agrawala, *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet), ICCV 2023. <https://arxiv.org/abs/2302.05543>
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), 2021. <https://arxiv.org/abs/2103.00020>
- Models on Hugging Face: [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), [control_v11p_sd15_scribble](https://huggingface.co/lllyasviel/control_v11p_sd15_scribble), [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
- ControlNet v1.1: <https://github.com/lllyasviel/ControlNet-v1-1-nightly>
- Libraries: [PyTorch](https://pytorch.org), [Hugging Face Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), [Accelerate](https://github.com/huggingface/accelerate), [OpenCV](https://opencv.org), [Gradio](https://www.gradio.app), pandas, matplotlib, Pillow.
- Sketches: drawn procedurally with `make_sketches.py` (author's own script); no external images or datasets are used.

## AI assistance
The code and documentation were developed with the help of an AI assistant (Claude, Anthropic). The author reviewed, ran and is able to explain the implementation.
