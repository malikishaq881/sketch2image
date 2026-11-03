"""Generate four example line-drawing sketches in sketches/ (wobbly, pencil-like strokes).

The sketches are drawn procedurally; they are not scanned from paper. Replace them with your own
hand-drawn sketches if you prefer: any PNG/JPG with dark lines on a light background works.

    python make_sketches.py
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

CANVAS = 1024          # drawn at 1024 px, saved at 512 px
OUTPUT_SIZE = 512
OUTPUT_DIR = Path("sketches")
SEED = 7

Point = tuple[float, float]


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("L", (CANVAS, CANVAS), 245)
    return image, ImageDraw.Draw(image)


def line(draw: ImageDraw.ImageDraw, p: Point, q: Point, width: int = 5) -> None:
    """Draw a wobbly line (two slightly different strokes, like a pencil)."""
    for _ in range(2):
        segments = max(int(math.dist(p, q) / 18), 2)
        jitter = 3
        points = [
            (
                p[0] + (q[0] - p[0]) * i / segments + random.uniform(-jitter, jitter),
                p[1] + (q[1] - p[1]) * i / segments + random.uniform(-jitter, jitter),
            )
            for i in range(segments + 1)
        ]
        draw.line(points, fill=random.randint(20, 60), width=width, joint="curve")


def polygon(draw: ImageDraw.ImageDraw, points: list[Point], closed: bool = True) -> None:
    edges = list(zip(points, points[1:] + ([points[0]] if closed else [])))
    for a, b in edges:
        line(draw, a, b)


def rectangle(draw: ImageDraw.ImageDraw, x0: float, y0: float, x1: float, y1: float) -> None:
    polygon(draw, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def ellipse(draw: ImageDraw.ImageDraw, cx: float, cy: float, rx: float, ry: float) -> None:
    for _ in range(2):
        offset = random.uniform(0, 6.28)
        points = [
            (
                cx + rx * math.cos(t + offset) + random.uniform(-3, 3),
                cy + ry * math.sin(t + offset) + random.uniform(-3, 3),
            )
            for t in [i * 6.4 / 40 for i in range(41)]
        ]
        draw.line(points, fill=random.randint(20, 60), width=5, joint="curve")


def save(image: Image.Image, name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    image = image.filter(ImageFilter.GaussianBlur(0.8))
    image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS).save(OUTPUT_DIR / f"{name}.png")


def building() -> None:
    image, d = new_canvas()
    rectangle(d, 380, 260, 640, 900)
    polygon(d, [(380, 260), (510, 150), (640, 260)])
    line(d, (510, 150), (510, 70))
    for y in range(300, 880, 80):
        for x in (410, 480, 550):
            rectangle(d, x, y, x + 50, y + 45)
    line(d, (250, 900), (780, 900))
    rectangle(d, 300, 780, 380, 900)
    rectangle(d, 640, 760, 730, 900)
    save(image, "building")


def robot() -> None:
    image, d = new_canvas()
    rectangle(d, 380, 160, 640, 360)
    ellipse(d, 450, 250, 32, 32)
    ellipse(d, 570, 250, 32, 32)
    line(d, (440, 320), (580, 320))
    line(d, (510, 160), (510, 90))
    ellipse(d, 510, 75, 18, 18)
    rectangle(d, 340, 380, 680, 720)
    rectangle(d, 230, 400, 330, 660)
    rectangle(d, 690, 400, 790, 660)
    rectangle(d, 400, 740, 490, 930)
    rectangle(d, 530, 740, 620, 930)
    ellipse(d, 510, 550, 50, 50)
    save(image, "robot")


def car() -> None:
    image, d = new_canvas()
    polygon(d, [(140, 640), (200, 540), (380, 470), (620, 460), (800, 540), (880, 600), (890, 660)],
            closed=False)
    line(d, (140, 640), (890, 660))
    line(d, (380, 470), (330, 560))
    line(d, (620, 460), (690, 560))
    line(d, (330, 560), (690, 560))
    line(d, (480, 490), (490, 560))
    for cx, cy in ((290, 670), (720, 680)):
        ellipse(d, cx, cy, 90, 90)
        ellipse(d, cx, cy, 40, 40)
    line(d, (100, 780), (930, 780))
    save(image, "car")


def rocket() -> None:
    image, d = new_canvas()
    polygon(d, [(430, 700), (430, 330), (510, 150), (590, 330), (590, 700)])
    ellipse(d, 510, 380, 45, 45)
    line(d, (430, 560), (590, 560))
    polygon(d, [(430, 620), (320, 780), (430, 740)])
    polygon(d, [(590, 620), (700, 780), (590, 740)])
    polygon(d, [(450, 700), (510, 880), (570, 700)], closed=False)
    polygon(d, [(480, 720), (510, 810), (540, 720)], closed=False)
    for _ in range(6):  # stars
        ellipse(d, random.randint(100, 900), random.randint(80, 300), 5, 5)
    save(image, "rocket")


if __name__ == "__main__":
    random.seed(SEED)
    for draw_sketch in (building, robot, car, rocket):
        draw_sketch()
    print(f"Saved 4 sketches in {OUTPUT_DIR}/")
