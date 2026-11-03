"""Run the sketch-to-image experiments on every sketch in a folder.

Example:
    python run_experiments.py --sketches sketches --prompts prompts.json --out results
    python run_experiments.py --quick        # small sweeps for a fast smoke test

For every sketch the script saves the control map, all generated images and one comparison grid
per experiment. Globally it saves results.csv (all runs), summary.csv (means), overview.png,
plot_cond_scale.png and run_info.json (settings and library versions).
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from config import (  # noqa: E402
    BASE_MODEL,
    CLIP_MODEL,
    CONTROLNET_MODEL,
    NEGATIVE_PROMPT,
    STYLES,
    GenerationSettings,
    build_prompt,
)
from metrics import ClipScorer, sketch_preservation  # noqa: E402
from pipeline import generate, load_pipeline, preprocess_sketch  # noqa: E402

logger = logging.getLogger("experiments")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
METRIC_COLUMNS = ["recall", "precision", "iou", "chamfer", "clip"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sketches", default="sketches", help="folder with input sketches")
    parser.add_argument("--prompts", default=None, help="JSON file: sketch name -> subject description")
    parser.add_argument("--out", default="results", help="output folder")
    parser.add_argument("--style", default="photorealistic", choices=list(STYLES),
                        help="style used in the parameter sweeps")
    parser.add_argument("--quick", action="store_true", help="smaller sweeps for a fast test")
    return parser.parse_args()


def build_sweeps(quick: bool) -> dict[str, list[tuple[str, dict]]]:
    """Experiments as {name: [(label, overrides), ...]}. Overrides may contain 'style'."""
    values = {
        "cond_scale": [0.5, 1.0, 1.5] if quick else [0.3, 0.6, 1.0, 1.3, 1.6],
        "guidance": [] if quick else [5.0, 7.5, 12.0],
        "steps": [] if quick else [15, 30, 50],
        "seed": [1, 2] if quick else [1, 2, 3, 4],
    }
    sweeps: dict[str, list[tuple[str, dict]]] = {"style": [(s, {"style": s}) for s in STYLES]}
    for name, items in values.items():
        if items:
            sweeps[name] = [(f"{name}_{v}", {name: v}) for v in items]
    return sweeps


def save_grid(rows: list[list[Image.Image]], column_labels: list[str], path: Path,
              cell: int = 256, header: int = 24) -> None:
    """Paste images into a labelled grid (one list per row) and save it."""
    columns = max(len(r) for r in rows)
    grid = Image.new("RGB", (cell * columns, header + cell * len(rows)), "white")
    draw = ImageDraw.Draw(grid)
    for c, label in enumerate(column_labels[:columns]):
        draw.text((c * cell + 6, 7), label, fill="black")
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            grid.paste(image.resize((cell, cell)), (c * cell, header + r * cell))
    grid.save(path)


def environment_info(args: argparse.Namespace) -> dict:
    """Settings and library versions, saved for reproducibility."""
    packages = {}
    for name in ["torch", "diffusers", "transformers", "accelerate", "opencv-python-headless", "gradio"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "n/a"
    return {
        "python": platform.python_version(),
        "packages": packages,
        "models": {"base": BASE_MODEL, "controlnet": CONTROLNET_MODEL, "clip": CLIP_MODEL},
        "defaults": asdict(GenerationSettings()),
        "negative_prompt": NEGATIVE_PROMPT,
        "arguments": vars(args),
    }


def run_sketch(sketch_path: Path, subject: str, pipe, clip: ClipScorer, out_dir: Path,
               base_style: str, sweeps: dict[str, list[tuple[str, dict]]]
               ) -> tuple[list[dict], list[Image.Image]]:
    """Run every experiment on one sketch. Returns (records, overview row)."""
    name = sketch_path.stem
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)

    with Image.open(sketch_path) as raw:
        control = preprocess_sketch(raw)
    control.save(folder / "control.png")

    records: list[dict] = []
    overview_row: list[Image.Image] = [control]
    for experiment, variants in sweeps.items():
        images = []
        for label, overrides in variants:
            style = overrides.get("style", base_style)
            settings = replace(GenerationSettings(), **{k: v for k, v in overrides.items() if k != "style"})
            prompt = build_prompt(subject, style)

            image = generate(pipe, control, prompt, settings)
            image.save(folder / f"{experiment}_{label}.png")
            images.append(image)

            records.append({
                "sketch": name, "experiment": experiment, "variant": label, "style": style,
                "prompt": prompt, **asdict(settings),
                **sketch_preservation(control, image), "clip": clip(image, prompt),
            })
            logger.info("%s | %s | recall=%.2f clip=%.1f", name, label,
                        records[-1]["recall"], records[-1]["clip"])

        save_grid([[control, *images]], ["sketch", *[label for label, _ in variants]],
                  folder / f"grid_{experiment}.png")
        if experiment == "style":
            overview_row += images
    return records, overview_row


def plot_cond_scale(df: pd.DataFrame, path: Path) -> None:
    """Plot the effect of the ControlNet conditioning scale on the metrics."""
    data = df[df.experiment == "cond_scale"].groupby("cond_scale")[["recall", "iou", "clip"]].mean()
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    data[["recall", "iou"]].plot(ax=axes[0], marker="o", title="Sketch preservation")
    data[["clip"]].plot(ax=axes[1], marker="o", color="tab:green", title="Prompt fidelity (CLIP score)")
    for axis in axes:
        axis.set_xlabel("ControlNet conditioning scale")
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sketches = sorted(p for p in Path(args.sketches).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not sketches:
        raise SystemExit(f"No sketches found in '{args.sketches}'")
    subjects = json.loads(Path(args.prompts).read_text()) if args.prompts else {}

    (out_dir / "run_info.json").write_text(json.dumps(environment_info(args), indent=2))

    pipe = load_pipeline()
    clip = ClipScorer()
    sweeps = build_sweeps(args.quick)

    records: list[dict] = []
    overview: list[list[Image.Image]] = []
    for path in sketches:
        subject = subjects.get(path.stem, path.stem.replace("_", " ").replace("-", " "))
        logger.info("=== %s: '%s' ===", path.stem, subject)
        sketch_records, row = run_sketch(path, subject, pipe, clip, out_dir, args.style, sweeps)
        records += sketch_records
        overview.append(row)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "results.csv", index=False)
    summary = df.groupby(["experiment", "variant"])[METRIC_COLUMNS].mean().round(3)
    summary.to_csv(out_dir / "summary.csv")
    plot_cond_scale(df, out_dir / "plot_cond_scale.png")
    save_grid(overview, ["sketch", *STYLES], out_dir / "overview.png")

    print("\nMean metrics per experiment:\n", summary.to_string())
    print(f"\nDone. Results saved in '{out_dir}/'")


if __name__ == "__main__":
    main()
