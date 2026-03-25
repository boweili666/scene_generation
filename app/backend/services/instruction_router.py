from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple


ROUTER_MODES = {"graph", "placement", "both", "reset"}

RESET_WORDS = (
    "clear scene",
    "start over",
    "new scene",
    "reset scene",
    "reset",
    "重新开始",
    "从头开始",
    "重新生成",
    "重置",
    "清空",
)
CREATE_WORDS = (
    "add",
    "append",
    "create",
    "insert",
    "place a",
    "place an",
    "new ",
    "新增",
    "添加",
    "加一个",
    "加一把",
    "放一个",
    "来一个",
    "创建",
)
DELETE_WORDS = (
    "delete",
    "drop",
    "remove",
    "删掉",
    "删除",
    "去掉",
    "移除",
)
REPLACE_WORDS = (
    "replace",
    "swap",
    "switch",
    "换成",
    "替换成",
    "改成",
)
MOVE_WORDS = (
    "move",
    "shift",
    "slide",
    "translate",
    "移动",
    "挪",
    "平移",
    "左移",
    "右移",
    "前移",
    "后移",
    "抬高",
    "降低",
    "往左",
    "往右",
    "往前",
    "往后",
)
ROTATE_WORDS = (
    "rotate",
    "turn",
    "yaw",
    "旋转",
    "转动",
    "转向",
    "朝向",
)
RELATION_WORDS = (
    "left of",
    "right of",
    "in front of",
    "behind",
    "against wall",
    "in corner",
    "on top of",
    "next to",
    "beside",
    "facing",
    "左边",
    "右边",
    "前面",
    "后面",
    "靠墙",
    "角落",
    "上面",
    "旁边",
    "对着",
)
SUPPORT_PHRASES = (
    "put on",
    "place on",
    "set on",
    "放到",
    "放在",
    "摆到",
    "摆在",
    "放上",
)
NUMERIC_UNITS = (
    "cm",
    "centimeter",
    "centimeters",
    "meter",
    "meters",
    "m ",
    "degree",
    "degrees",
    "deg",
    "厘米",
    "米",
    "度",
    "°",
)
SOFT_LAYOUT_WORDS = (
    "clean up the layout",
    "make it look better",
    "make it more natural",
    "tidy the layout",
    "整理一下布局",
    "摆整齐",
    "更自然一点",
    "更合理一点",
)

COMMON_CLASS_ALIASES = {
    "bed": ("床",),
    "book": ("书",),
    "chair": ("椅子",),
    "cup": ("杯子",),
    "desk": ("书桌", "桌子"),
    "lamp": ("灯", "台灯"),
    "mug": ("马克杯", "杯子"),
    "nightstand": ("床头柜",),
    "plant": ("植物", "盆栽"),
    "shelf": ("架子", "置物架"),
    "sofa": ("沙发",),
    "table": ("桌子",),
    "tv": ("电视",),
}

ASCII_TOKEN_RE = re.compile(r"^[a-z0-9_./-]+$")
SPLIT_TOKEN_RE = re.compile(r"[^a-z0-9_./-]+")
NUMERIC_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:cm|centimeter|centimeters|m|meter|meters|degree|degrees|deg|厘米|米|度|°)",
    re.IGNORECASE,
)


def build_router_rulebook() -> str:
    return "\n".join(
        [
            "Routing rules:",
            "- reset: clear or regenerate the whole scene.",
            "- graph: object set or semantic relations change.",
            "- placement: only coordinates/orientation of existing objects change.",
            "- both: relation and geometry both need an update.",
            "- If the instruction adds, removes, replaces, or renames objects, route graph.",
            "- If the instruction changes relative relations like left/right/on/against wall, route both.",
            "- If the instruction only changes numeric move/rotation of existing objects, route placement.",
            "- If uncertain, choose both instead of placement.",
        ]
    )


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    return any(phrase in text for phrase in phrases)


def _ascii_phrase_in_text(text: str, token: str) -> bool:
    pattern = rf"(^|[^a-z0-9_]){re.escape(token)}([^a-z0-9_]|$)"
    return re.search(pattern, text) is not None


def _text_mentions_token(text: str, token: str) -> bool:
    token = token.strip().lower()
    if not token:
        return False
    if ASCII_TOKEN_RE.fullmatch(token):
        return _ascii_phrase_in_text(text, token)
    return token in text


def normalize_instruction_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _object_tokens(path: str, meta: Dict[str, Any]) -> List[str]:
    tokens = {path.lower(), path.split("/")[-1].lower()}
    class_name = str(meta.get("class") or meta.get("class_name") or "").strip().lower()
    caption = str(meta.get("caption") or "").strip().lower()
    if class_name:
        tokens.add(class_name)
        for alias in COMMON_CLASS_ALIASES.get(class_name, ()):
            tokens.add(alias.lower())
    if caption:
        tokens.add(caption)
        for token in SPLIT_TOKEN_RE.split(caption):
            if len(token) >= 2:
                tokens.add(token)
    return sorted(token for token in tokens if token)


def resolve_mentioned_objects(text: str, scene_graph: Dict[str, Any] | None) -> List[str]:
    if not scene_graph:
        return []
    normalized = normalize_instruction_text(text)
    mentioned: List[str] = []
    for path, meta in (scene_graph.get("obj") or {}).items():
        if not isinstance(meta, dict):
            continue
        tokens = _object_tokens(path, meta)
        if any(_text_mentions_token(normalized, token) for token in tokens):
            mentioned.append(path)
    return sorted(dict.fromkeys(mentioned))


def collect_instruction_signals(
    text: str,
    scene_graph: Dict[str, Any] | None,
    placements: Dict[str, List[float]] | None,
) -> Dict[str, Any]:
    normalized = normalize_instruction_text(text)
    mentioned = resolve_mentioned_objects(normalized, scene_graph)
    has_reset = _contains_any(normalized, RESET_WORDS)
    has_create = _contains_any(normalized, CREATE_WORDS)
    has_delete = _contains_any(normalized, DELETE_WORDS)
    has_replace = _contains_any(normalized, REPLACE_WORDS)
    has_move = _contains_any(normalized, MOVE_WORDS)
    has_rotate = _contains_any(normalized, ROTATE_WORDS)
    has_relation = _contains_any(normalized, RELATION_WORDS)
    has_support_phrase = _contains_any(normalized, SUPPORT_PHRASES) or bool(
        re.search(r"\b(?:put|place|set)\b.*\bon\b", normalized)
    )
    has_soft_layout = _contains_any(normalized, SOFT_LAYOUT_WORDS)
    has_numeric_unit = bool(NUMERIC_UNIT_RE.search(normalized)) or _contains_any(normalized, NUMERIC_UNITS)
    placements_available = bool(placements)

    scores = {
        "reset": 0,
        "graph": 0,
        "placement": 0,
        "both": 0,
    }
    if has_reset:
        scores["reset"] += 100
    if has_create:
        scores["graph"] += 80
    if has_delete:
        scores["graph"] += 85
    if has_replace:
        scores["graph"] += 60
    if has_move:
        scores["placement"] += 45
    if has_rotate:
        scores["placement"] += 45
    if has_numeric_unit:
        scores["placement"] += 25
    if has_relation:
        scores["graph"] += 15
        scores["both"] += 45
    if has_support_phrase:
        scores["both"] += 60
    if has_soft_layout:
        scores["both"] += 20
    if mentioned and (has_move or has_rotate) and not has_create and not has_delete:
        scores["placement"] += 20
    if len(mentioned) >= 2 and has_relation:
        scores["both"] += 30

    active_signals = [
        name
        for name, enabled in (
            ("reset", has_reset),
            ("create", has_create),
            ("delete", has_delete),
            ("replace", has_replace),
            ("move", has_move),
            ("rotate", has_rotate),
            ("relation", has_relation),
            ("support", has_support_phrase),
            ("numeric", has_numeric_unit),
            ("soft_layout", has_soft_layout),
        )
        if enabled
    ]
    return {
        "text": normalized,
        "mentioned_paths": mentioned,
        "mentioned_existing_count": len(mentioned),
        "placements_available": placements_available,
        "has_reset": has_reset,
        "has_create": has_create,
        "has_delete": has_delete,
        "has_replace": has_replace,
        "has_move": has_move,
        "has_rotate": has_rotate,
        "has_relation": has_relation,
        "has_support_phrase": has_support_phrase,
        "has_numeric_unit": has_numeric_unit,
        "has_soft_layout": has_soft_layout,
        "signals": active_signals,
        "scores": scores,
    }


def _coerce_route_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in ROUTER_MODES:
        return mode
    return ""


def _fallback_mode(signals: Dict[str, Any]) -> Tuple[str, float, str]:
    if signals["has_reset"]:
        return "reset", 0.99, "matched reset keywords"
    if signals["has_create"] or signals["has_delete"] or signals["has_replace"]:
        return "graph", 0.95, "matched object create/delete/replace keywords"
    if signals["has_support_phrase"]:
        return "both", 0.92, "matched support relation phrase"
    if signals["has_relation"] and signals["mentioned_existing_count"] >= 2:
        return "both", 0.88, "matched relation keywords with multiple existing objects"
    if (signals["has_move"] or signals["has_rotate"]) and signals["has_numeric_unit"] and not signals["has_relation"]:
        return "placement", 0.86, "matched numeric move/rotate keywords"

    scores = signals["scores"]
    mode = max(scores, key=scores.get)
    if scores[mode] <= 0:
        return "both", 0.5, "no strong rule match; defaulted to both"
    return mode, 0.64, "fell back to score-based routing"


def validate_route_decision(
    text: str,
    scene_graph: Dict[str, Any] | None,
    placements: Dict[str, List[float]] | None,
    llm_route: Dict[str, Any] | None,
) -> Dict[str, Any]:
    signals = collect_instruction_signals(text, scene_graph, placements)
    fallback_mode, fallback_confidence, fallback_reason = _fallback_mode(signals)
    llm_mode = _coerce_route_mode((llm_route or {}).get("mode"))
    try:
        llm_confidence = float((llm_route or {}).get("confidence", 0.0))
    except (TypeError, ValueError):
        llm_confidence = 0.0
    llm_reason = str((llm_route or {}).get("reason") or "").strip()

    mode = fallback_mode
    confidence = fallback_confidence
    reason_bits = [fallback_reason]

    hard_override = fallback_mode if fallback_confidence >= 0.85 else ""
    if hard_override:
        reason_bits = [f"backend override: {fallback_reason}"]
    elif llm_mode and llm_confidence >= 0.55:
        mode = llm_mode
        confidence = max(0.55, min(llm_confidence, 0.99))
        reason_bits = [llm_reason or "accepted llm router decision"]

    if mode == "placement" and (
        signals["has_create"]
        or signals["has_delete"]
        or signals["has_replace"]
    ):
        mode = "graph"
        confidence = max(confidence, 0.9)
        reason_bits.append("placement rejected because object set changes")

    if mode == "placement" and (signals["has_relation"] or signals["has_support_phrase"]) and signals["mentioned_existing_count"] >= 2:
        mode = "both"
        confidence = max(confidence, 0.9)
        reason_bits.append("placement rejected because semantic relation changes")

    requires_existing_placements = mode == "placement" and not signals["placements_available"]
    if requires_existing_placements:
        reason_bits.append("placement edits need existing sampled placements")

    needs_resample = mode in {"both", "reset"} or signals["has_relation"] or signals["has_support_phrase"]
    apply_placement_editor = mode == "placement" or (
        mode == "both"
        and signals["placements_available"]
        and (signals["has_move"] or signals["has_rotate"])
    )

    return {
        "mode": mode,
        "confidence": round(confidence, 2),
        "signals": signals["signals"],
        "objects": signals["mentioned_paths"],
        "reason": "; ".join(bit for bit in reason_bits if bit),
        "needs_resample": needs_resample,
        "requires_existing_placements": requires_existing_placements,
        "apply_placement_editor": apply_placement_editor,
    }


def prune_placements_to_scene_graph(
    scene_graph: Dict[str, Any],
    placements: Dict[str, List[float]] | None,
) -> Dict[str, List[float]]:
    valid_paths = set((scene_graph or {}).get("obj") or {})
    pruned: Dict[str, List[float]] = {}
    for path, payload in (placements or {}).items():
        if path in valid_paths:
            pruned[path] = list(payload)
    return pruned


def _edge_signatures(scene_graph: Dict[str, Any] | None) -> Dict[Tuple[str, str], set[str]]:
    edges = ((scene_graph or {}).get("edges") or {}).get("obj-obj") or []
    signatures: Dict[Tuple[str, str], set[str]] = {}
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        relation = edge.get("relation")
        if not source or not target or not relation:
            continue
        signatures.setdefault((str(source), str(target)), set()).add(str(relation))
    return signatures


def reconcile_placements_after_graph_edit(
    previous_graph: Dict[str, Any],
    next_graph: Dict[str, Any],
    placements: Dict[str, List[float]] | None,
    route: Dict[str, Any] | None,
) -> tuple[Dict[str, List[float]], List[str]]:
    current = prune_placements_to_scene_graph(next_graph, placements)
    previous_objects = (previous_graph or {}).get("obj") or {}
    next_objects = (next_graph or {}).get("obj") or {}

    previous_paths = set(previous_objects)
    next_paths = set(next_objects)
    new_paths = next_paths - previous_paths
    changed_paths = {
        path
        for path in (previous_paths & next_paths)
        if previous_objects.get(path) != next_objects.get(path)
    }

    previous_edges = _edge_signatures(previous_graph)
    next_edges = _edge_signatures(next_graph)
    edge_affected: set[str] = set()
    for key in set(previous_edges) | set(next_edges):
        if previous_edges.get(key, set()) != next_edges.get(key, set()):
            edge_affected.update(key)

    invalidate_paths = set(route.get("objects") or [])
    invalidate_paths.update(new_paths)
    invalidate_paths.update(changed_paths)
    invalidate_paths.update(edge_affected)

    reconciled = dict(current)
    for path in invalidate_paths:
        reconciled.pop(path, None)

    return reconciled, sorted(invalidate_paths)
