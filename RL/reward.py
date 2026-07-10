"""Reward helpers for RL training.

Two layered reward variants share the same invalid branch:

  is_valid := graph penalty three components all zero (no invalid edges /
              cycles / subgraphs); format also OK if format_penalty_weight > 0.

  if not is_valid:
      reward = -graph_penalty_sum * INVALID_PENALTY_SCALE          # range ~ [-30, 0]

The valid branch differs:

  * compute_reward_iou (current default) — IoU vs GT open lines:
      reward = VALID_BASE + IOU_WEIGHT * iou                       # range [VALID_BASE, VALID_BASE+IOU_WEIGHT]

  * compute_reward_full (legacy, sim-based) — pandapower improvement:
      reward = VALID_BASE + clip(impr_ratio, -CAP, +CAP) * SIM_W   # range ~ [+1, +16]

Invariant for both: VALID_BASE - max_negative_bonus  >  0  >=  -graph_penalty_sum * INVALID_PENALTY_SCALE
                    ⇒ any valid reward > any invalid reward.
                    ⇒ model cannot game the system by deliberately producing invalid configs.

The IoU variant has no simulator dependency, so RL training does not need
pandapower. Eval still uses the simulator independently for diagnostics.
"""

from __future__ import annotations

from utils.metrics_utils import (
    compute_gt_match,
    graph_penalties,
    graph_penalties_from_open_lines,
    parse_open_lines,
    parse_open_lines_full_xml,
    parse_open_lines_xml,
)


def _has_full_format(response: str) -> bool:
    return (
        bool(parse_open_lines(response))
        and ("Node Voltages=" in response or "NodeVoltages=" in response)
        and "System Loss=" in response
    )


def compute_reward(
    prompt: str,
    response: str,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """Legacy graph-only reward (kept for tests / backwards compatibility)."""
    parts = dict(graph_penalties(prompt, response))
    parts["format_penalty"] = 0.0 if _has_full_format(response) else 1.0

    reward = -(
        invalid_edges_weight * parts["invalid_edges"]
        + cycles_weight * parts["cycles"]
        + subgraphs_weight * parts["subgraphs"]
        + format_penalty_weight * parts["format_penalty"]
    )
    return float(reward), parts


def compute_reward_full(
    prompt: str,
    response: str,
    *,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    sim_weight: float = 50.0,
    valid_base: float = 6.0,
    sim_neg_cap: float = 0.10,
    sim_pos_cap: float = 0.20,
    invalid_penalty_scale: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """Full layered reward (graph validity + simulator-derived improvement).

    Guarantees `valid_reward_min > invalid_reward_max` so the model cannot
    fall back to invalid configs to escape negative sim bonuses.
    """
    parts = dict(graph_penalties(prompt, response))
    parts["format_penalty"] = 0.0 if _has_full_format(response) else 1.0

    is_valid = (
        parts["invalid_edges"] == 0.0
        and parts["cycles"] == 0.0
        and parts["subgraphs"] == 0.0
        and (format_penalty_weight <= 0 or parts["format_penalty"] == 0.0)
    )
    parts["is_valid"] = 1.0 if is_valid else 0.0
    parts["sim_bonus"] = 0.0
    parts["sim_converged"] = 0.0

    if not is_valid:
        graph_penalty_sum = (
            invalid_edges_weight * parts["invalid_edges"]
            + cycles_weight * parts["cycles"]
            + subgraphs_weight * parts["subgraphs"]
            + format_penalty_weight * parts["format_penalty"]
        )
        reward = -graph_penalty_sum * invalid_penalty_scale
        return float(reward), parts

    # valid branch — simulator-based sim_bonus, clipped to keep invariant.
    sim_bonus = 0.0
    if sim_weight > 0:
        try:
            from RL.simulator.grid_simulator import evaluate_reconfig

            open_lines = parse_open_lines(response)
            if open_lines:
                result = evaluate_reconfig(prompt, open_lines)
                orig = result.get("original_loss_mw") or 0.0
                if result.get("converged") and orig > 0:
                    impr_ratio = result["improvement_mw"] / orig
                    impr_ratio = max(-sim_neg_cap, min(sim_pos_cap, impr_ratio))
                    sim_bonus = impr_ratio * sim_weight
                    parts["sim_converged"] = 1.0
        except Exception:
            pass  # simulator unavailable or errored — sim_bonus stays 0

    parts["sim_bonus"] = float(sim_bonus)
    reward = valid_base + sim_bonus
    return float(reward), parts


def compute_reward_valid_only(
    prompt: str,
    response: str,
    *,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    valid_base: float = 1.0,
    invalid_penalty_scale: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """Only reward valid topology/format; no GT IoU and no simulator."""
    parts = dict(graph_penalties(prompt, response))
    parts["format_penalty"] = 0.0 if _has_full_format(response) else 1.0
    is_valid = (
        parts["invalid_edges"] == 0.0
        and parts["cycles"] == 0.0
        and parts["subgraphs"] == 0.0
        and (format_penalty_weight <= 0 or parts["format_penalty"] == 0.0)
    )
    parts["is_valid"] = 1.0 if is_valid else 0.0

    if is_valid:
        return float(valid_base), parts

    graph_penalty_sum = (
        invalid_edges_weight * parts["invalid_edges"]
        + cycles_weight * parts["cycles"]
        + subgraphs_weight * parts["subgraphs"]
        + format_penalty_weight * parts["format_penalty"]
    )
    return float(-graph_penalty_sum * invalid_penalty_scale), parts


def compute_reward_iou_only(
    prompt: str,
    response: str,
    gt_response: str,
    *,
    iou_weight: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """Only reward Open Lines IoU vs GT; graph penalties are diagnostics."""
    parts = dict(graph_penalties(prompt, response))
    parts["format_penalty"] = 0.0 if _has_full_format(response) else 1.0

    gen_open = parse_open_lines(response)
    gt_open = parse_open_lines(gt_response) if gt_response else []
    exact, iou = compute_gt_match(gen_open, gt_open)
    parts["iou"] = float(iou)
    parts["gt_exact_match"] = float(exact)

    return float(iou_weight * iou), parts


def _has_xml_format(response: str) -> bool:
    return bool(parse_open_lines_xml(response))


def _has_full_xml_format(response: str) -> bool:
    return bool(parse_open_lines_full_xml(response))


def _canonical_edge_set(lines) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for line in lines or []:
        if not isinstance(line, (tuple, list)) or len(line) != 2:
            continue
        try:
            a, b = int(line[0]), int(line[1])
        except (TypeError, ValueError):
            continue
        edges.add((min(a, b), max(a, b)))
    return edges


def _edge_set_iou(predicted, correct) -> tuple[float, float]:
    pred_set = _canonical_edge_set(predicted)
    correct_set = _canonical_edge_set(correct)
    exact = 1.0 if pred_set == correct_set else 0.0
    union = pred_set | correct_set
    iou = (len(pred_set & correct_set) / len(union)) if union else 1.0
    return exact, iou


def _edge_precision_recall(predicted, correct) -> tuple[float, float]:
    pred_set = _canonical_edge_set(predicted)
    correct_set = _canonical_edge_set(correct)
    intersection = pred_set & correct_set
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(correct_set) if correct_set else 0.0
    return precision, recall


def compute_reward_xml_valid_only(
    prompt: str,
    response: str,
    *,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    valid_base: float = 1.0,
    invalid_penalty_scale: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """XML-only graph validity reward for <answer><open_lines>...</open_lines></answer>."""
    return _compute_reward_xml_valid_only_with_parser(
        prompt=prompt,
        response=response,
        parse_xml_open_lines=parse_open_lines_xml,
        has_xml_format=_has_xml_format,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
    )


def _compute_reward_xml_valid_only_with_parser(
    prompt: str,
    response: str,
    *,
    parse_xml_open_lines,
    has_xml_format,
    invalid_edges_weight: float,
    cycles_weight: float,
    subgraphs_weight: float,
    format_penalty_weight: float,
    valid_base: float,
    invalid_penalty_scale: float,
) -> tuple[float, dict[str, float]]:
    gen_open = parse_xml_open_lines(response)
    parts = dict(graph_penalties_from_open_lines(prompt, gen_open))
    parts["format_penalty"] = 0.0 if has_xml_format(response) else 1.0
    is_valid = (
        parts["invalid_edges"] == 0.0
        and parts["cycles"] == 0.0
        and parts["subgraphs"] == 0.0
        and (format_penalty_weight <= 0 or parts["format_penalty"] == 0.0)
    )
    parts["is_valid"] = 1.0 if is_valid else 0.0

    if is_valid:
        return float(valid_base), parts

    graph_penalty_sum = (
        invalid_edges_weight * parts["invalid_edges"]
        + cycles_weight * parts["cycles"]
        + subgraphs_weight * parts["subgraphs"]
        + format_penalty_weight * parts["format_penalty"]
    )
    return float(-graph_penalty_sum * invalid_penalty_scale), parts


def compute_reward_xml_iou_only(
    prompt: str,
    response: str,
    gt_response: str,
    *,
    iou_weight: float = 8.0,
    improvement_weight: float = 2.0,
    precision_weight: float = 1.0,
    recall_weight: float = 1.0,
    copy_penalty: float = 2.0,
) -> tuple[float, dict[str, float]]:
    """XML-only GT Open Lines reward with current-config improvement shaping."""
    return _compute_reward_xml_iou_only_with_parser(
        prompt=prompt,
        response=response,
        gt_response=gt_response,
        parse_xml_open_lines=parse_open_lines_xml,
        has_xml_format=_has_xml_format,
        iou_weight=iou_weight,
        improvement_weight=improvement_weight,
        precision_weight=precision_weight,
        recall_weight=recall_weight,
        copy_penalty=copy_penalty,
    )


def _compute_reward_xml_iou_only_with_parser(
    prompt: str,
    response: str,
    gt_response: str,
    *,
    parse_xml_open_lines,
    has_xml_format,
    iou_weight: float,
    improvement_weight: float,
    precision_weight: float,
    recall_weight: float,
    copy_penalty: float,
) -> tuple[float, dict[str, float]]:
    gen_open = parse_xml_open_lines(response)
    gt_open = parse_xml_open_lines(gt_response) if gt_response else []
    current_open = parse_open_lines(prompt)
    parts = dict(graph_penalties_from_open_lines(prompt, gen_open))
    parts["format_penalty"] = 0.0 if has_xml_format(response) else 1.0

    exact, iou_pred_gt = _edge_set_iou(gen_open, gt_open)
    _, iou_current_gt = _edge_set_iou(current_open, gt_open)
    edge_precision, edge_recall = _edge_precision_recall(gen_open, gt_open)

    pred_set = _canonical_edge_set(gen_open)
    gt_set = _canonical_edge_set(gt_open)
    current_set = _canonical_edge_set(current_open)
    copied_current = pred_set == current_set and gt_set != current_set
    applied_copy_penalty = copy_penalty if copied_current else 0.0

    parts["iou"] = float(iou_pred_gt)
    parts["iou_current_gt"] = float(iou_current_gt)
    parts["edge_precision"] = float(edge_precision)
    parts["edge_recall"] = float(edge_recall)
    parts["gt_exact_match"] = float(exact)
    parts["copied_current_open"] = 1.0 if copied_current else 0.0
    parts["copy_penalty"] = float(applied_copy_penalty)

    reward = (
        iou_weight * iou_pred_gt
        + improvement_weight * max(0.0, iou_pred_gt - iou_current_gt)
        + precision_weight * edge_precision
        + recall_weight * edge_recall
        - applied_copy_penalty
    )
    return float(reward), parts



def compute_reward_xml_valid_and_iou(
    prompt: str,
    response: str,
    gt_response: str,
    *,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    iou_weight: float = 8.0,
    improvement_weight: float = 2.0,
    precision_weight: float = 1.0,
    recall_weight: float = 1.0,
    copy_penalty: float = 2.0,
    valid_base: float = 0.0,
    invalid_penalty_scale: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """XML-only graph-valid gate plus GT Open Lines IoU."""
    return _compute_reward_xml_valid_and_iou_with_parser(
        prompt=prompt,
        response=response,
        gt_response=gt_response,
        parse_xml_open_lines=parse_open_lines_xml,
        has_xml_format=_has_xml_format,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        iou_weight=iou_weight,
        improvement_weight=improvement_weight,
        precision_weight=precision_weight,
        recall_weight=recall_weight,
        copy_penalty=copy_penalty,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
    )


def compute_reward_full_xml_valid_and_iou(
    prompt: str,
    response: str,
    gt_response: str,
    *,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    iou_weight: float = 8.0,
    improvement_weight: float = 2.0,
    precision_weight: float = 1.0,
    recall_weight: float = 1.0,
    copy_penalty: float = 2.0,
    valid_base: float = 0.0,
    invalid_penalty_scale: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """Full XML graph-valid gate plus GT Open Lines IoU.

    This mirrors compute_reward_xml_valid_and_iou for ablations, but the format
    gate requires open_lines, node_voltages, and system_loss XML fields.
    """
    return _compute_reward_xml_valid_and_iou_with_parser(
        prompt=prompt,
        response=response,
        gt_response=gt_response,
        parse_xml_open_lines=parse_open_lines_full_xml,
        has_xml_format=_has_full_xml_format,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        iou_weight=iou_weight,
        improvement_weight=improvement_weight,
        precision_weight=precision_weight,
        recall_weight=recall_weight,
        copy_penalty=copy_penalty,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
    )


def _compute_reward_xml_valid_and_iou_with_parser(
    prompt: str,
    response: str,
    gt_response: str,
    *,
    parse_xml_open_lines,
    has_xml_format,
    invalid_edges_weight: float,
    cycles_weight: float,
    subgraphs_weight: float,
    format_penalty_weight: float,
    iou_weight: float,
    improvement_weight: float,
    precision_weight: float,
    recall_weight: float,
    copy_penalty: float,
    valid_base: float,
    invalid_penalty_scale: float,
) -> tuple[float, dict[str, float]]:
    valid_reward, parts = _compute_reward_xml_valid_only_with_parser(
        prompt=prompt,
        response=response,
        parse_xml_open_lines=parse_xml_open_lines,
        has_xml_format=has_xml_format,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
    )
    parts["iou"] = 0.0
    parts["iou_current_gt"] = 0.0
    parts["edge_precision"] = 0.0
    parts["edge_recall"] = 0.0
    parts["gt_exact_match"] = 0.0
    parts["copied_current_open"] = 0.0
    parts["copy_penalty"] = 0.0
    if parts["is_valid"] == 0.0:
        return valid_reward, parts

    iou_reward, iou_parts = _compute_reward_xml_iou_only_with_parser(
        prompt=prompt,
        response=response,
        gt_response=gt_response,
        parse_xml_open_lines=parse_xml_open_lines,
        has_xml_format=has_xml_format,
        iou_weight=iou_weight,
        improvement_weight=improvement_weight,
        precision_weight=precision_weight,
        recall_weight=recall_weight,
        copy_penalty=copy_penalty,
    )
    parts["iou"] = iou_parts["iou"]
    parts["iou_current_gt"] = iou_parts["iou_current_gt"]
    parts["edge_precision"] = iou_parts["edge_precision"]
    parts["edge_recall"] = iou_parts["edge_recall"]
    parts["gt_exact_match"] = iou_parts["gt_exact_match"]
    parts["copied_current_open"] = iou_parts["copied_current_open"]
    parts["copy_penalty"] = iou_parts["copy_penalty"]
    return float(valid_base + iou_reward), parts


def compute_reward_valid_and_iou(
    prompt: str,
    response: str,
    gt_response: str,
    *,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    iou_weight: float = 10.0,
    valid_base: float = 1.0,
    invalid_penalty_scale: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """Current default shape: invalid penalty; valid gets base + IoU bonus."""
    valid_reward, parts = compute_reward_valid_only(
        prompt=prompt,
        response=response,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
    )
    parts["iou"] = 0.0
    parts["gt_exact_match"] = 0.0
    if parts["is_valid"] == 0.0:
        return valid_reward, parts

    iou_reward, iou_parts = compute_reward_iou_only(
        prompt=prompt,
        response=response,
        gt_response=gt_response,
        iou_weight=iou_weight,
    )
    parts["iou"] = iou_parts["iou"]
    parts["gt_exact_match"] = iou_parts["gt_exact_match"]
    return float(valid_base + iou_reward), parts


def group_advantages(rewards: list[float]) -> list[float]:
    mean_reward = sum(rewards) / len(rewards)
    advantages = [reward - mean_reward for reward in rewards]
    variance = sum(value * value for value in advantages) / len(advantages)
    if variance > 1e-8:
        scale = variance**0.5
        advantages = [value / scale for value in advantages]
    return advantages


def compute_reward_iou(
    prompt: str,
    response: str,
    gt_response: str,
    *,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    iou_weight: float = 10.0,
    valid_base: float = 1.0,
    invalid_penalty_scale: float = 10.0,
) -> tuple[float, dict[str, float]]:
    """Layered reward: graph validity gate + IoU-vs-GT bonus on the valid branch.

    No simulator dependency. The IoU is computed by ``compute_gt_match`` over
    undirected canonical edges (same definition used by eval).

    Reward shape (with defaults valid_base=1, iou_weight=10, scale=10):
      invalid sample : -graph_penalty_sum * 10        ≈ [-30, 0]
      valid + iou=0  : 1                             (worst valid)
      valid + iou=1  : 11                            (perfect match)
    """
    return compute_reward_valid_and_iou(
        prompt=prompt,
        response=response,
        gt_response=gt_response,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        iou_weight=iou_weight,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
    )
