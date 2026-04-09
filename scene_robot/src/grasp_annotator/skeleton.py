from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from skimage import morphology
from skimage.draw import disk as draw_disk
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize

from .render import load_font
from .segmentation import extract_contour, segment_object


@dataclass
class GraspCandidate:
    id: int
    center_xy: Tuple[int, int]
    gripper_axis_xy: Tuple[float, float]
    tangent_xy: Tuple[float, float]
    width_px: float
    score: float
    branch_id: int
    candidate_type: str


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v, dtype=float)
    return v / n


def angle_of(v: np.ndarray) -> float:
    return math.atan2(v[1], v[0])


def inside_image(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_skeleton_and_distance(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cleaned = morphology.binary_closing(mask, morphology.disk(3))
    cleaned = morphology.binary_dilation(cleaned, morphology.disk(1))
    cleaned = morphology.binary_closing(cleaned, morphology.disk(2))
    cleaned = morphology.remove_small_objects(cleaned, 256)
    dist = ndi.distance_transform_edt(cleaned)
    skeleton = skeletonize(cleaned)
    skeleton = morphology.binary_closing(skeleton, morphology.disk(1))
    skeleton = skeletonize(skeleton)
    return skeleton, dist


def reconnect_nearby_skeleton_gaps(
    skel: np.ndarray,
    max_gap_px: float = 10.0,
    min_component_size: int = 12,
) -> np.ndarray:
    labels, num = ndi.label(skel)
    if num <= 1:
        return skel

    component_sizes = ndi.sum(skel, labels, index=np.arange(1, num + 1))
    large_labels = {
        idx + 1
        for idx, size in enumerate(component_sizes)
        if float(size) >= float(min_component_size)
    }
    if len(large_labels) <= 1:
        return skel

    out = skel.copy()
    endpoints, _ = extract_graph_nodes(out)
    endpoints_by_label: dict[int, list[tuple[int, int]]] = {}
    for x, y in endpoints:
        lab = int(labels[y, x])
        if lab in large_labels:
            endpoints_by_label.setdefault(lab, []).append((x, y))

    best_links: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int], float]] = {}
    large_list = sorted(large_labels)
    for i, lab_a in enumerate(large_list):
        pts_a = endpoints_by_label.get(lab_a, [])
        if not pts_a:
            continue
        for lab_b in large_list[i + 1:]:
            pts_b = endpoints_by_label.get(lab_b, [])
            if not pts_b:
                continue
            best = None
            for ax, ay in pts_a:
                for bx, by in pts_b:
                    gap = math.hypot(ax - bx, ay - by)
                    if gap <= max_gap_px and (best is None or gap < best[2]):
                        best = ((ax, ay), (bx, by), gap)
            if best is not None:
                best_links[(lab_a, lab_b)] = best

    for (ax, ay), (bx, by), _ in best_links.values():
        rr, cc = draw_line(ay, ax, by, bx)
        out[rr, cc] = True

    out = morphology.binary_closing(out, morphology.disk(1))
    out = skeletonize(out)
    return out


def neighbor_count_map(skel: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    conv = ndi.convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    return np.where(skel, conv - 10, 0)


def prune_skeleton_by_distance(skel: np.ndarray, dist: np.ndarray, min_dist_px: float = 2.0) -> np.ndarray:
    out = skel.copy()
    out[(skel) & (dist < min_dist_px)] = False
    return out


def get_skeleton_pixels(skel: np.ndarray) -> List[Tuple[int, int]]:
    ys, xs = np.where(skel)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def skeleton_neighbors(skel: np.ndarray, x: int, y: int) -> List[Tuple[int, int]]:
    h, w = skel.shape
    nbrs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            xx = x + dx
            yy = y + dy
            if 0 <= xx < w and 0 <= yy < h and skel[yy, xx]:
                nbrs.append((xx, yy))
    return nbrs


def extract_graph_nodes(skel: np.ndarray) -> Tuple[set, set]:
    counts = neighbor_count_map(skel)
    ys, xs = np.where(skel)
    endpoints = set()
    junctions = set()
    for x, y in zip(xs, ys):
        c = counts[y, x]
        if c == 1:
            endpoints.add((int(x), int(y)))
        elif c >= 3:
            junctions.add((int(x), int(y)))
    return endpoints, junctions


def trace_branches(skel: np.ndarray) -> List[List[Tuple[int, int]]]:
    endpoints, junctions = extract_graph_nodes(skel)
    special = endpoints | junctions
    visited_edges = set()
    branches = []

    def edge_key(a, b):
        return tuple(sorted([a, b]))

    for node in special:
        x0, y0 = node
        for nbr in skeleton_neighbors(skel, x0, y0):
            ek = edge_key(node, nbr)
            if ek in visited_edges:
                continue

            branch = [node]
            prev = node
            cur = nbr
            visited_edges.add(ek)

            while True:
                branch.append(cur)
                if cur in special and cur != node:
                    break

                nbrs = skeleton_neighbors(skel, cur[0], cur[1])
                nxts = [p for p in nbrs if p != prev]
                if len(nxts) == 0:
                    break
                if len(nxts) > 1:
                    break

                nxt = nxts[0]
                ek = edge_key(cur, nxt)
                if ek in visited_edges:
                    break
                visited_edges.add(ek)
                prev, cur = cur, nxt

            if len(branch) >= 2:
                branches.append(branch)

    unlabeled = skel.copy()
    for br in branches:
        for x, y in br:
            unlabeled[y, x] = False

    labels, num = ndi.label(unlabeled)
    for lab in range(1, num + 1):
        ys, xs = np.where(labels == lab)
        pts = list(zip(xs.tolist(), ys.tolist()))
        if len(pts) >= 8:
            branches.append([(int(x), int(y)) for x, y in pts])

    return branches


def branch_length(branch: List[Tuple[int, int]]) -> float:
    if len(branch) < 2:
        return 0.0
    length = 0.0
    for i in range(1, len(branch)):
        x0, y0 = branch[i - 1]
        x1, y1 = branch[i]
        length += math.hypot(x1 - x0, y1 - y0)
    return length


def prune_short_branches(skel: np.ndarray, branches: List[List[Tuple[int, int]]], min_branch_len: float = 18.0) -> np.ndarray:
    pruned = np.zeros_like(skel, dtype=bool)
    for br in branches:
        if branch_length(br) >= min_branch_len:
            for x, y in br:
                pruned[y, x] = True
    return pruned


def estimate_local_tangent(branch: List[Tuple[int, int]], idx: int, window: int = 5) -> np.ndarray:
    i0 = max(0, idx - window)
    i1 = min(len(branch) - 1, idx + window)
    x0, y0 = branch[i0]
    x1, y1 = branch[i1]
    v = np.array([x1 - x0, y1 - y0], dtype=float)
    return normalize(v)


def sample_indices_evenly(n: int, k: int, margin: int = 4) -> List[int]:
    if n <= 2 * margin + 1:
        return []
    valid = np.arange(margin, n - margin)
    if len(valid) == 0:
        return []
    if k >= len(valid):
        return valid.tolist()
    idxs = np.linspace(0, len(valid) - 1, k).round().astype(int)
    return valid[np.unique(idxs)].tolist()


def branch_curvature_penalty(branch: List[Tuple[int, int]], idx: int, step: int = 4) -> float:
    i0 = max(0, idx - step)
    i1 = min(len(branch) - 1, idx + step)
    if i1 <= i0 + 1:
        return 0.5

    xm, ym = branch[idx]
    x0, y0 = branch[i0]
    x1, y1 = branch[i1]

    v0 = normalize(np.array([xm - x0, ym - y0], dtype=float))
    v1 = normalize(np.array([x1 - xm, y1 - ym], dtype=float))
    dot = float(np.clip(np.dot(v0, v1), -1.0, 1.0))
    return (dot + 1.0) / 2.0


def point_to_nearest_contour_distance(contour_xy: np.ndarray, x: int, y: int) -> float:
    pts = contour_xy.astype(float)
    d = np.sqrt((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
    return float(np.min(d)) if len(d) else 0.0


def generate_grasp_candidates(
    mask: np.ndarray,
    skeleton: np.ndarray,
    dist: np.ndarray,
    max_candidates: int = 24,
    min_width_px: float = 12.0,
    max_width_px: float = 220.0,
) -> List[GraspCandidate]:
    branches = trace_branches(skeleton)
    branches = [br for br in branches if branch_length(br) >= 18.0]
    contour_xy = extract_contour(mask)
    candidates: List[GraspCandidate] = []
    cid = 0

    if not branches:
        return candidates

    max_dist_global = float(np.max(dist[skeleton])) if np.any(skeleton) else 1.0
    max_dist_global = max(max_dist_global, 1.0)
    lengths = np.array([branch_length(br) for br in branches], dtype=float)
    total_len = max(float(np.sum(lengths)), 1.0)

    for bid, branch in enumerate(branches):
        n_branch = max(2, int(round(max_candidates * (branch_length(branch) / total_len))))
        idxs = sample_indices_evenly(len(branch), n_branch, margin=5)
        for idx in idxs:
            x, y = branch[idx]
            width_px = float(dist[y, x] * 2.0)
            if width_px < min_width_px or width_px > max_width_px:
                continue

            tangent = estimate_local_tangent(branch, idx, window=5)
            if np.linalg.norm(tangent) < 1e-6:
                continue

            axis = normalize(np.array([-tangent[1], tangent[0]], dtype=float))
            centrality = clamp01(dist[y, x] / max_dist_global)
            straightness = branch_curvature_penalty(branch, idx, step=4)
            d_contour = point_to_nearest_contour_distance(contour_xy, x, y)
            contour_factor = clamp01(d_contour / (width_px * 0.4 + 1e-6))
            score = 0.50 * centrality + 0.30 * straightness + 0.20 * contour_factor

            candidates.append(
                GraspCandidate(
                    id=cid,
                    center_xy=(int(x), int(y)),
                    gripper_axis_xy=(float(axis[0]), float(axis[1])),
                    tangent_xy=(float(tangent[0]), float(tangent[1])),
                    width_px=width_px,
                    score=float(score),
                    branch_id=bid,
                    candidate_type="centerline",
                )
            )
            cid += 1

    candidates.sort(key=lambda c: c.score, reverse=True)
    filtered: List[GraspCandidate] = []
    min_sep = 18.0
    for c in candidates:
        keep = True
        for f in filtered:
            dx = c.center_xy[0] - f.center_xy[0]
            dy = c.center_xy[1] - f.center_xy[1]
            if math.hypot(dx, dy) < min_sep:
                keep = False
                break
        if keep:
            filtered.append(c)
        if len(filtered) >= max_candidates:
            break
    return filtered


def analyze_grasp_image(
    rgb: np.ndarray,
    *,
    max_candidates: int = 24,
    min_width_px: float = 12.0,
    max_width_px: float = 220.0,
    min_dist_px: float = 1.25,
    min_branch_len: float = 10.0,
    max_gap_px: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, list[GraspCandidate]]:
    mask = segment_object(rgb)
    skeleton, dist = compute_skeleton_and_distance(mask)
    skeleton = reconnect_nearby_skeleton_gaps(skeleton, max_gap_px=max_gap_px)
    skeleton = prune_skeleton_by_distance(skeleton, dist, min_dist_px=min_dist_px)
    branches = trace_branches(skeleton)
    skeleton = prune_short_branches(skeleton, branches, min_branch_len=min_branch_len)
    candidates = generate_grasp_candidates(
        mask=mask,
        skeleton=skeleton,
        dist=dist,
        max_candidates=max_candidates,
        min_width_px=min_width_px,
        max_width_px=max_width_px,
    )
    return mask, skeleton, candidates


def overlay_skeleton(rgb: np.ndarray, skeleton: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[mask] = (0.75 * out[mask] + 0.25 * np.array([255, 245, 200])).astype(np.uint8)
    thick = morphology.dilation(skeleton, morphology.disk(1))
    out[thick] = np.array([220, 30, 30], dtype=np.uint8)
    return out


def draw_candidate_marks(
    image: Image.Image,
    candidates: List[GraspCandidate],
    line_scale: float = 0.45,
    show_ids: bool = True,
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = load_font(18)
    w, h = out.size

    for c in candidates:
        x, y = c.center_xy
        axis = np.array(c.gripper_axis_xy, dtype=float)
        half = max(8, int(round(c.width_px * line_scale / 2.0)))
        p0 = np.array([x, y], dtype=float) - axis * half
        p1 = np.array([x, y], dtype=float) + axis * half
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))

        draw.line((x0, y0, x1, y1), fill=(40, 220, 40), width=3)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 100, 0), outline=(255, 255, 255), width=1)

        if show_ids:
            label = str(c.id)
            l, t, r, b = draw.textbbox((0, 0), label, font=font)
            tw, th = r - l, b - t
            tx = min(max(x + 8, 2), w - tw - 4)
            ty = min(max(y - th - 8, 2), h - th - 4)
            draw.rounded_rectangle((tx - 2, ty - 1, tx + tw + 2, ty + th + 1), radius=4, fill=(255, 255, 255))
            draw.text((tx, ty), label, fill=(200, 30, 30), font=font)

    return out


def draw_candidates_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    skeleton: np.ndarray,
    candidates: List[GraspCandidate],
    line_scale: float = 0.45,
) -> np.ndarray:
    out = overlay_skeleton(rgb, skeleton, mask)
    h, w, _ = out.shape
    for c in candidates:
        x, y = c.center_xy
        axis = np.array(c.gripper_axis_xy, dtype=float)
        half = max(8, int(round(c.width_px * line_scale / 2.0)))
        p0 = np.array([x, y], dtype=float) - axis * half
        p1 = np.array([x, y], dtype=float) + axis * half
        x0, y0 = int(round(p0[0])), int(round(p0[1]))
        x1, y1 = int(round(p1[0])), int(round(p1[1]))

        rr, cc = draw_line(y0, x0, y1, x1)
        valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        out[rr[valid], cc[valid]] = np.array([30, 200, 30], dtype=np.uint8)
        rr2, cc2 = draw_disk((y, x), radius=4, shape=(h, w))
        out[rr2, cc2] = np.array([255, 80, 0], dtype=np.uint8)
    return out


def save_candidates_json(candidates: List[GraspCandidate], path: Path) -> None:
    data = {"num_candidates": len(candidates), "candidates": [asdict(c) for c in candidates]}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_grasp_candidate_artifacts(
    image: Image.Image,
    mask: np.ndarray,
    skeleton: np.ndarray,
    candidates: List[GraspCandidate],
    output_prefix: Path,
    *,
    review_prefix: Path | None = None,
) -> dict[str, Path]:
    rgb = np.array(image.convert("RGB"))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if review_prefix is None:
        review_prefix = output_prefix
    review_prefix.parent.mkdir(parents=True, exist_ok=True)

    mask_path = output_prefix.parent / f"{output_prefix.name}_mask.png"
    skeleton_path = output_prefix.parent / f"{output_prefix.name}_skeleton.png"
    overlay_path = output_prefix.parent / f"{output_prefix.name}_skeleton_overlay.png"
    candidates_overlay_path = review_prefix.parent / f"{review_prefix.name}_grasp_candidates.png"
    candidates_on_original_path = review_prefix.parent / f"{review_prefix.name}_grasp_candidates_on_original.png"
    candidates_json_path = output_prefix.parent / f"{output_prefix.name}_grasp_candidates.json"

    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
    Image.fromarray((skeleton.astype(np.uint8) * 255), mode="L").save(skeleton_path)
    Image.fromarray(overlay_skeleton(rgb, skeleton, mask), mode="RGB").save(overlay_path)
    Image.fromarray(draw_candidates_overlay(rgb, mask, skeleton, candidates), mode="RGB").save(candidates_overlay_path)
    draw_candidate_marks(image, candidates, show_ids=True).save(candidates_on_original_path)
    save_candidates_json(candidates, candidates_json_path)

    return {
        "mask_path": mask_path,
        "skeleton_path": skeleton_path,
        "overlay_path": overlay_path,
        "candidates_overlay_path": candidates_overlay_path,
        "candidates_on_original_path": candidates_on_original_path,
        "candidates_json_path": candidates_json_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2D grasp candidates from a single rendered image.")
    parser.add_argument("--input", type=Path, required=True, help="Input PNG/JPG image")
    parser.add_argument("--out-mask", type=Path, default=None, help="Output binary mask PNG")
    parser.add_argument("--out-skeleton", type=Path, default=None, help="Output binary skeleton PNG")
    parser.add_argument("--out-overlay", type=Path, default=None, help="Output skeleton overlay PNG")
    parser.add_argument("--out-candidates-overlay", type=Path, default=None, help="Output candidate overlay PNG")
    parser.add_argument("--out-candidates-on-original", type=Path, default=None, help="Output candidate marks directly on original image")
    parser.add_argument("--out-json", type=Path, default=None, help="Output candidates JSON")
    parser.add_argument("--max-candidates", type=int, default=24, help="Max number of final candidates")
    parser.add_argument("--min-width-px", type=float, default=12.0, help="Reject grasp if local width is too small")
    parser.add_argument("--max-width-px", type=float, default=220.0, help="Reject grasp if local width is too large")
    parser.add_argument("--min-dist-px", type=float, default=1.25, help="Remove very thin skeleton parts")
    parser.add_argument("--min-branch-len", type=float, default=10.0, help="Remove short branches after reconnecting")
    parser.add_argument("--max-gap-px", type=float, default=10.0, help="Reconnect nearby skeleton gaps up to this size")
    args = parser.parse_args()

    image = Image.open(args.input).convert("RGB")
    rgb = np.array(image)
    mask, skeleton_map, candidates = analyze_grasp_image(
        rgb,
        max_candidates=args.max_candidates,
        min_width_px=args.min_width_px,
        max_width_px=args.max_width_px,
        min_dist_px=args.min_dist_px,
        min_branch_len=args.min_branch_len,
        max_gap_px=args.max_gap_px,
    )

    stem = args.input.with_suffix("")
    out_mask = args.out_mask or Path(f"{stem}_mask.png")
    out_skeleton = args.out_skeleton or Path(f"{stem}_skeleton.png")
    out_overlay = args.out_overlay or Path(f"{stem}_skeleton_overlay.png")
    out_candidates_overlay = args.out_candidates_overlay or Path(f"{stem}_grasp_candidates.png")
    out_candidates_on_original = args.out_candidates_on_original or Path(f"{stem}_grasp_candidates_on_original.png")
    out_json = args.out_json or Path(f"{stem}_grasp_candidates.json")
    for path in (out_mask, out_skeleton, out_overlay, out_candidates_overlay, out_candidates_on_original, out_json):
        path.parent.mkdir(parents=True, exist_ok=True)

    prefix = stem.parent / stem.name
    artifacts = save_grasp_candidate_artifacts(image, mask, skeleton_map, candidates, prefix)
    if artifacts["mask_path"] != out_mask:
        artifacts["mask_path"].replace(out_mask)
    if artifacts["skeleton_path"] != out_skeleton:
        artifacts["skeleton_path"].replace(out_skeleton)
    if artifacts["overlay_path"] != out_overlay:
        artifacts["overlay_path"].replace(out_overlay)
    if artifacts["candidates_overlay_path"] != out_candidates_overlay:
        artifacts["candidates_overlay_path"].replace(out_candidates_overlay)
    if artifacts["candidates_on_original_path"] != out_candidates_on_original:
        artifacts["candidates_on_original_path"].replace(out_candidates_on_original)
    if artifacts["candidates_json_path"] != out_json:
        artifacts["candidates_json_path"].replace(out_json)

    print(f"Saved mask: {out_mask}")
    print(f"Saved skeleton: {out_skeleton}")
    print(f"Saved overlay: {out_overlay}")
    print(f"Saved candidates overlay: {out_candidates_overlay}")
    print(f"Saved candidates on original: {out_candidates_on_original}")
    print(f"Saved candidates json: {out_json}")
    print(f"Generated {len(candidates)} candidates.")


if __name__ == "__main__":
    main()
