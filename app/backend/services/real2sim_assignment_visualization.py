from __future__ import annotations

import html
import json
import os
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .real2sim_review_service import load_assignment_review


ASSIGNMENT_BBOX_OVERLAY_FILENAME = "assignment_bbox_overlay.png"
ASSIGNMENT_REVIEW_HTML_FILENAME = "assignment_review.html"
_MASK_PALETTE = [
    (230, 57, 70),
    (29, 78, 216),
    (46, 125, 50),
    (255, 140, 0),
    (111, 66, 193),
    (0, 121, 107),
    (214, 51, 132),
    (120, 144, 156),
]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _resolve_candidate(base_dir: Path, raw_path: Any) -> Path | None:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _resolve_assignment_inputs(assignment_path: Path) -> dict[str, Path | None]:
    payload = _read_json(assignment_path)
    results_dir = assignment_path.parent.resolve()

    scene_graph_path = _resolve_candidate(results_dir, payload.get("scene_graph_path"))
    if scene_graph_path is None or not scene_graph_path.exists():
        raise FileNotFoundError(f"Scene graph referenced by assignment.json was not found: {scene_graph_path}")

    review_image = _resolve_candidate(results_dir, payload.get("image_path"))
    masks_dir = review_image.parent if review_image is not None else None

    if masks_dir is None or not masks_dir.exists():
        for row in payload.get("mask_labels", []):
            if not isinstance(row, dict):
                continue
            mask_path = _resolve_candidate(results_dir, row.get("mask_path"))
            if mask_path is not None and mask_path.exists():
                masks_dir = mask_path.parent
                break

    if masks_dir is None or not masks_dir.exists():
        default_masks_dir = results_dir.parent / "masks"
        if default_masks_dir.exists():
            masks_dir = default_masks_dir

    if masks_dir is None or not masks_dir.exists():
        raise FileNotFoundError(f"Could not infer masks directory for {assignment_path}")

    return {
        "scene_graph_path": scene_graph_path.resolve(),
        "masks_dir": masks_dir.resolve(),
        "results_dir": results_dir,
        "latest_input_image": review_image.resolve() if review_image is not None and review_image.exists() else None,
    }


def _coerce_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x_min, y_min, x_max, y_max = [int(round(float(v))) for v in value]
    except (TypeError, ValueError):
        return None
    if x_max < x_min or y_max < y_min:
        return None
    return x_min, y_min, x_max, y_max


def _load_font(image_size: tuple[int, int]) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    target_size = max(16, min(image_size) // 22)
    return _load_font_with_size(target_size)


def _load_font_with_size(target_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for font_name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, target_size)
        except OSError:
            continue
    return ImageFont.load_default()


def _color_for_mask(mask_label: int) -> tuple[int, int, int]:
    return _MASK_PALETTE[(mask_label - 1) % len(_MASK_PALETTE)]


def _basename(scene_path: str | None) -> str:
    value = str(scene_path or "").strip()
    return value.rsplit("/", 1)[-1] if value else ""


def _measure_multiline_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    spacing: int,
) -> tuple[int, int]:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _rectangles_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _place_badge(
    bbox: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
    badge_size: tuple[int, int],
    used_rects: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    x_min, y_min, x_max, y_max = bbox
    badge_width, badge_height = badge_size
    candidate_origins = [
        (x_min + 8, y_min - badge_height - 10),
        (x_min + 8, y_min + 8),
        (x_max - badge_width - 8, y_min - badge_height - 10),
        (x_max - badge_width - 8, y_min + 8),
        (x_min + 8, y_max - badge_height - 8),
    ]

    def clamp(origin: tuple[int, int]) -> tuple[int, int, int, int]:
        left = max(0, min(origin[0], image_size[0] - badge_width))
        top = max(0, min(origin[1], image_size[1] - badge_height))
        return left, top, left + badge_width, top + badge_height

    for step in range(6):
        for candidate in candidate_origins:
            shifted = (candidate[0], candidate[1] + step * (badge_height + 6))
            rect = clamp(shifted)
            if not any(_rectangles_intersect(rect, used) for used in used_rects):
                used_rects.append(rect)
                return rect[0], rect[1]

    rect = clamp(candidate_origins[0])
    used_rects.append(rect)
    return rect[0], rect[1]


def _mask_status(mask_label: int, assignment_by_mask: dict[int, dict[str, Any]]) -> str:
    return "matched" if mask_label in assignment_by_mask else "unmatched"


def _badge_label(mask_label: int, assignment: dict[str, Any] | None) -> str:
    if assignment is None:
        return f"M{mask_label}\nunmatched"
    target = _basename(assignment.get("scene_path")) or "assigned"
    return f"M{mask_label}\n{target}"


def _legend_rows(review: dict[str, Any]) -> list[dict[str, Any]]:
    assignment_by_mask = {
        int(row["mask_label"]): row
        for row in review.get("assignments", [])
        if isinstance(row, dict) and isinstance(row.get("mask_label"), int)
    }
    rows: list[dict[str, Any]] = []
    for mask in review.get("mask_labels", []):
        if not isinstance(mask, dict) or not isinstance(mask.get("mask_label"), int):
            continue
        mask_label = int(mask["mask_label"])
        assignment = assignment_by_mask.get(mask_label)
        status = _mask_status(mask_label, assignment_by_mask)
        confidence = f"{float(assignment.get('confidence') or 0.0) * 100:.0f}%" if assignment is not None else "n/a"
        target = _basename(assignment.get("scene_path")) if assignment is not None else "Unassigned"
        prompt = str(mask.get("prompt") or "n/a").strip() or "n/a"
        scene_class = str(assignment.get("scene_class") or "") if assignment is not None else ""
        meta_parts = [f"prompt {prompt}"]
        if assignment is not None:
            meta_parts.append(f"conf {confidence}")
        if scene_class:
            meta_parts.append(scene_class)
        rows.append(
            {
                "mask_label": mask_label,
                "output_name": str(mask.get("output_name") or ""),
                "status": status,
                "target": target,
                "meta": " • ".join(meta_parts),
                "color": _color_for_mask(mask_label),
            }
        )
    return rows


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    fill: tuple[int, int, int, int],
    max_width: int,
    spacing: int = 4,
) -> int:
    if not text.strip():
        return 0
    font_size = max(12, int(getattr(font, "size", 16)))
    wrap_width = max(12, int(max_width / max(7, font_size * 0.58)))
    wrapped = "\n".join(textwrap.wrap(text, width=wrap_width, break_long_words=False, break_on_hyphens=False)) or text
    draw.multiline_text(xy, wrapped, font=font, fill=fill, spacing=spacing)
    _, height = _measure_multiline_text(draw, text=wrapped, font=font, spacing=spacing)
    return height


def render_assignment_bbox_overlay(review: dict[str, Any], *, output_path: str | Path) -> Path:
    review_image = Path(str(review.get("review_image_abs_path") or ""))
    if not review_image.exists():
        raise FileNotFoundError(f"Review image not found: {review_image}")

    output_file = Path(output_path)
    base = Image.open(review_image).convert("RGBA")
    overlay = base.copy()
    badge_font = _load_font_with_size(max(16, min(base.size) // 42))
    image_draw = ImageDraw.Draw(overlay, "RGBA")
    border_width = max(3, min(base.size) // 550)
    used_badge_rects: list[tuple[int, int, int, int]] = []

    assignment_by_mask = {
        int(row["mask_label"]): row
        for row in review.get("assignments", [])
        if isinstance(row, dict) and isinstance(row.get("mask_label"), int)
    }

    for mask in review.get("mask_labels", []):
        if not isinstance(mask, dict):
            continue
        mask_label = mask.get("mask_label")
        if not isinstance(mask_label, int):
            continue
        bbox = _coerce_bbox(mask.get("bbox_xyxy"))
        if bbox is None:
            continue

        assignment = assignment_by_mask.get(mask_label)
        color = _color_for_mask(mask_label)
        x_min, y_min, x_max, y_max = bbox
        outline_alpha = 240 if assignment is not None else 170
        image_draw.rectangle((x_min, y_min, x_max, y_max), outline=(*color, outline_alpha), width=border_width)

        label_text = _badge_label(mask_label, assignment)
        text_width, text_height = _measure_multiline_text(image_draw, text=label_text, font=badge_font, spacing=3)
        padding_x = 10
        padding_y = 6
        badge_width = text_width + padding_x * 2
        badge_height = text_height + padding_y * 2
        box_left, box_top = _place_badge(
            bbox,
            image_size=base.size,
            badge_size=(badge_width, badge_height),
            used_rects=used_badge_rects,
        )
        anchor_x = min(max(x_min + 14, box_left + badge_width // 2), x_max)
        anchor_y = min(max(y_min + 14, box_top + badge_height // 2), y_max)
        image_draw.line(
            (box_left + badge_width // 2, box_top + badge_height // 2, anchor_x, anchor_y),
            fill=(*color, 210),
            width=max(2, border_width - 1),
        )
        image_draw.rounded_rectangle(
            (box_left, box_top, box_left + badge_width, box_top + badge_height),
            radius=10,
            fill=(*color, 236),
            outline=(255, 255, 255, 245),
            width=2,
        )
        image_draw.text(
            (box_left + padding_x, box_top + padding_y),
            label_text,
            fill=(255, 255, 255, 255),
            font=badge_font,
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_file)
    return output_file


def _relative_href(source_dir: Path, target: Path | None) -> str | None:
    if target is None:
        return None
    try:
        rel = os.path.relpath(target.resolve(), source_dir.resolve())
        return Path(rel).as_posix()
    except ValueError:
        return target.resolve().as_uri()


def _optional_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value)


def _format_bbox(mask: dict[str, Any]) -> str:
    bbox = _coerce_bbox(mask.get("bbox_xyxy"))
    if bbox is None:
        return "n/a"
    return ", ".join(str(value) for value in bbox)


def _render_chip_list(values: list[str | int], empty_text: str) -> str:
    if not values:
        return f'<div class="empty-note">{html.escape(empty_text)}</div>'
    chips = "".join(f'<span class="chip">{html.escape(str(value))}</span>' for value in values)
    return f'<div class="chip-list">{chips}</div>'


def _render_review_html(
    review: dict[str, Any],
    *,
    assignment_path: Path,
    html_output_path: Path,
    bbox_output_path: Path,
) -> str:
    summary = review.get("summary", {}) if isinstance(review.get("summary"), dict) else {}
    manifest = review.get("manifest", {}) if isinstance(review.get("manifest"), dict) else {}
    assignment_by_mask = {
        int(row["mask_label"]): row
        for row in review.get("assignments", [])
        if isinstance(row, dict) and isinstance(row.get("mask_label"), int)
    }
    html_dir = html_output_path.parent.resolve()
    review_image_href = _relative_href(html_dir, _optional_path(review.get("review_image_abs_path"))) or ""
    numbered_overlay_href = _relative_href(html_dir, _optional_path(review.get("overlay_image_abs_path"))) or ""
    bbox_overlay_href = _relative_href(html_dir, bbox_output_path.resolve()) or ""

    table_rows: list[str] = []
    for mask in review.get("mask_labels", []):
        if not isinstance(mask, dict) or not isinstance(mask.get("mask_label"), int):
            continue
        assignment = assignment_by_mask.get(int(mask["mask_label"]))
        status = "matched" if assignment is not None else "unmatched"
        confidence = f"{float(assignment.get('confidence') or 0.0) * 100:.0f}%" if assignment is not None else "n/a"
        scene_path = str(assignment.get("scene_path") or "Unassigned") if assignment is not None else "Unassigned"
        scene_class = str(assignment.get("scene_class") or "") if assignment is not None else ""
        scene_caption = str(assignment.get("scene_caption") or "") if assignment is not None else ""
        reason = str(assignment.get("reason") or "") if assignment is not None else ""
        prompt = str(mask.get("prompt") or "") or "n/a"
        mask_href = _relative_href(html_dir, Path(str(mask.get("mask_path")))) or "#"
        table_rows.append(
            f"""
            <tr class="{status}">
              <td><span class="status-pill {status}">{html.escape(status)}</span></td>
              <td class="mono">M{int(mask["mask_label"])}</td>
              <td class="mono">{html.escape(str(mask.get("output_name") or ""))}</td>
              <td>{html.escape(prompt)}</td>
              <td class="mono">{html.escape(_format_bbox(mask))}</td>
              <td class="mono">{html.escape(scene_path)}</td>
              <td>{html.escape(scene_class or "n/a")}</td>
              <td>{html.escape(scene_caption or "n/a")}</td>
              <td>{html.escape(confidence)}</td>
              <td>{html.escape(reason or "n/a")}</td>
              <td><a href="{html.escape(mask_href)}" target="_blank" rel="noreferrer">mask</a></td>
            </tr>
            """.strip()
        )

    source_label = "Needs attention" if review.get("needs_attention") else "Ready"
    source_class = "warn" if review.get("needs_attention") else "ok"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Real2Sim Assignment Review</title>
  <style>
    :root {{
      --bg: #f2ede4;
      --panel: rgba(255, 252, 246, 0.92);
      --panel-strong: #fff9ef;
      --ink: #20323f;
      --muted: #65737f;
      --line: rgba(46, 61, 74, 0.14);
      --accent: #b85a2b;
      --accent-soft: rgba(184, 90, 43, 0.12);
      --ok: #1e7a5c;
      --ok-soft: rgba(30, 122, 92, 0.14);
      --warn: #cb6f17;
      --warn-soft: rgba(203, 111, 23, 0.14);
      --shadow: 0 24px 60px rgba(38, 46, 58, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184, 90, 43, 0.12), transparent 28rem),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1560px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero {{
      display: grid;
      gap: 18px;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .hero-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.8rem, 3vw, 2.6rem);
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      color: var(--muted);
      margin: 6px 0 0;
      line-height: 1.6;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      font-size: 0.92rem;
      border: 1px solid var(--line);
      background: var(--panel-strong);
    }}
    .badge.ok {{
      color: var(--ok);
      background: var(--ok-soft);
      border-color: rgba(30, 122, 92, 0.22);
    }}
    .badge.warn {{
      color: var(--warn);
      background: var(--warn-soft);
      border-color: rgba(203, 111, 23, 0.22);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
    }}
    .stat {{
      padding: 16px 18px;
      border-radius: 18px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    .stat .value {{
      margin-top: 10px;
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.04em;
    }}
    .section {{
      margin-top: 22px;
      padding: 22px;
      border-radius: 24px;
      border: 1px solid var(--line);
      background: var(--panel);
      box-shadow: var(--shadow);
    }}
    .section h2 {{
      margin: 0 0 16px;
      font-size: 1.2rem;
      letter-spacing: -0.03em;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    figure {{
      margin: 0;
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid var(--line);
      background: #fbf7f0;
    }}
    figure img {{
      display: block;
      width: 100%;
      height: auto;
      background: #efe8db;
    }}
    figcaption {{
      padding: 12px 14px;
      color: var(--muted);
      font-size: 0.92rem;
      border-top: 1px solid var(--line);
    }}
    .lists {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .note-card {{
      padding: 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
    }}
    .note-card h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .chip-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 0.92rem;
      font-family: "SFMono-Regular", Consolas, monospace;
    }}
    .empty-note {{
      color: var(--muted);
      line-height: 1.6;
    }}
    .table-wrap {{
      overflow-x: auto;
      border-radius: 20px;
      border: 1px solid var(--line);
      background: var(--panel-strong);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1080px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.94rem;
      line-height: 1.45;
    }}
    th {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      background: rgba(255, 248, 236, 0.8);
      position: sticky;
      top: 0;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .status-pill {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .status-pill.matched {{
      color: var(--ok);
      background: var(--ok-soft);
    }}
    .status-pill.unmatched {{
      color: var(--warn);
      background: var(--warn-soft);
    }}
    .mono {{
      font-family: "SFMono-Regular", Consolas, monospace;
      font-size: 0.88rem;
    }}
    .footer {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.6;
    }}
    a {{
      color: #1a5fb4;
    }}
    @media (max-width: 720px) {{
      main {{ padding: 16px; }}
      .hero, .section {{ padding: 16px; border-radius: 18px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="hero-head">
        <div>
          <h1>Real2Sim Assignment Review</h1>
          <p class="subtitle">Static visualization generated from <span class="mono">{html.escape(str(assignment_path.resolve()))}</span></p>
        </div>
        <div class="badge {source_class}">{html.escape(source_label)}</div>
      </div>
      <div class="meta-grid">
        <div class="stat"><div class="label">Matched Assignments</div><div class="value">{int(summary.get("matched_assignments", 0))}</div></div>
        <div class="stat"><div class="label">Scene Objects</div><div class="value">{int(summary.get("scene_objects", 0))}</div></div>
        <div class="stat"><div class="label">Mask Labels</div><div class="value">{int(summary.get("mask_labels", 0))}</div></div>
        <div class="stat"><div class="label">Unmatched Masks</div><div class="value">{int(summary.get("unmatched_mask_labels", 0))}</div></div>
        <div class="stat"><div class="label">Unmatched Scene Paths</div><div class="value">{int(summary.get("unmatched_scene_paths", 0))}</div></div>
        <div class="stat"><div class="label">Low Confidence</div><div class="value">{int(summary.get("low_confidence_assignments", 0))}</div></div>
      </div>
      <div class="footer">
        Source: <span class="mono">{html.escape(str(review.get("source") or "unknown"))}</span><br/>
        Scene graph: <span class="mono">{html.escape(str(review.get("scene_graph_path") or ""))}</span><br/>
        Manifest matched objects: <span class="mono">{int(manifest.get("matched_objects", 0))}</span>
      </div>
    </section>

    <section class="section">
      <h2>Overlays</h2>
      <div class="media-grid">
        <figure>
          <img src="{html.escape(review_image_href)}" alt="Review image" loading="lazy" />
          <figcaption>Original review image</figcaption>
        </figure>
        <figure>
          <img src="{html.escape(numbered_overlay_href)}" alt="Numbered mask overlay" loading="lazy" />
          <figcaption>Numbered mask overlay</figcaption>
        </figure>
        <figure>
          <img src="{html.escape(bbox_overlay_href)}" alt="Assignment bbox overlay" loading="lazy" />
          <figcaption>Assignment bbox overlay with direct per-mask labels</figcaption>
        </figure>
      </div>
    </section>

    <section class="section">
      <h2>Open Issues</h2>
      <div class="lists">
        <div class="note-card">
          <h3>Unmatched Scene Paths</h3>
          {_render_chip_list(review.get("unmatched_scene_paths", []), "Every scene node was matched.")}
        </div>
        <div class="note-card">
          <h3>Unmatched Mask Labels</h3>
          {_render_chip_list(review.get("unmatched_mask_labels", []), "Every mask label was assigned.")}
        </div>
        <div class="note-card">
          <h3>Manifest Unmatched Scene Paths</h3>
          {_render_chip_list(manifest.get("unmatched_scene_paths", []), "Manifest does not report unmatched scene nodes.")}
        </div>
        <div class="note-card">
          <h3>Manifest Unbound Outputs</h3>
          {_render_chip_list(manifest.get("unmatched_outputs", []), "Manifest does not report unbound outputs.")}
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Assignment Table</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Status</th>
              <th>Mask</th>
              <th>Output</th>
              <th>Prompt</th>
              <th>BBox</th>
              <th>Scene Path</th>
              <th>Scene Class</th>
              <th>Scene Caption</th>
              <th>Confidence</th>
              <th>Reason</th>
              <th>Mask File</th>
            </tr>
          </thead>
          <tbody>
            {"".join(table_rows)}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def build_assignment_visualization(
    assignment_path: str | Path,
    *,
    html_output_path: str | Path | None = None,
    bbox_output_path: str | Path | None = None,
) -> dict[str, Any]:
    assignment_file = Path(assignment_path).expanduser().resolve()
    if not assignment_file.exists():
        raise FileNotFoundError(f"Assignment JSON not found: {assignment_file}")

    inputs = _resolve_assignment_inputs(assignment_file)
    review = load_assignment_review(
        scene_graph_path=inputs["scene_graph_path"],
        masks_dir=inputs["masks_dir"],
        results_dir=inputs["results_dir"],
        latest_input_image=inputs["latest_input_image"],
    )

    bbox_file = (
        Path(bbox_output_path).expanduser().resolve()
        if bbox_output_path is not None
        else inputs["results_dir"] / ASSIGNMENT_BBOX_OVERLAY_FILENAME
    )
    html_file = (
        Path(html_output_path).expanduser().resolve()
        if html_output_path is not None
        else inputs["results_dir"] / ASSIGNMENT_REVIEW_HTML_FILENAME
    )

    render_assignment_bbox_overlay(review, output_path=bbox_file)
    html_payload = _render_review_html(
        review,
        assignment_path=assignment_file,
        html_output_path=html_file,
        bbox_output_path=bbox_file,
    )
    html_file.parent.mkdir(parents=True, exist_ok=True)
    html_file.write_text(html_payload, encoding="utf-8")

    return {
        "review": review,
        "assignment_path": str(assignment_file),
        "bbox_output_path": str(bbox_file),
        "html_output_path": str(html_file),
    }
