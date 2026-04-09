from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import measure, morphology

from .render import load_font


def fill_small_holes_only(mask: np.ndarray, max_hole_area: int = 2000) -> np.ndarray:
    inv = ~mask
    labeled, num = ndi.label(inv)
    if num == 0:
        return mask

    border_labels = set(np.unique(np.concatenate([
        labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1],
    ])))
    out = mask.copy()
    for lab in range(1, num + 1):
        if lab in border_labels:
            continue
        if np.sum(labeled == lab) <= max_hole_area:
            out[labeled == lab] = True
    return out


def segment_object(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    mask = (gray < 235) & (chroma < 40)
    mask = morphology.closing(mask, morphology.disk(3))
    mask = morphology.opening(mask, morphology.disk(1))
    mask = morphology.remove_small_objects(mask, 512)

    labels, count = ndi.label(mask)
    if count == 0:
        raise ValueError("No foreground component found in top view.")
    sizes = ndi.sum(mask, labels, index=np.arange(1, count + 1))
    mask = labels == (int(np.argmax(sizes)) + 1)
    mask = fill_small_holes_only(mask, max_hole_area=2000)
    mask = morphology.closing(mask, morphology.disk(2))
    return morphology.remove_small_objects(mask, 512)


def extract_contour(mask: np.ndarray) -> np.ndarray:
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return np.zeros((0, 2), dtype=float)
    contour = max(contours, key=len)
    return np.stack([contour[:, 1], contour[:, 0]], axis=1)


def contour_centroid(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot compute centroid from an empty mask.")
    return int(round(xs.mean())), int(round(ys.mean()))


def draw_center_and_contour(
    image: Image.Image,
    contour_xy: np.ndarray,
    center_xy: tuple[int, int],
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = load_font(22)

    if len(contour_xy) > 1:
        points = [(float(x), float(y)) for x, y in contour_xy]
        draw.line(points + [points[0]], fill=(30, 180, 220), width=3)

    cx, cy = center_xy
    draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(255, 80, 40), outline=(255, 255, 255), width=2)
    draw.text((cx + 12, cy - 28), "center", fill=(255, 80, 40), font=font)
    return out
