"""veRL reward adapter for RL4DistReconfig."""

from __future__ import annotations

import os

from RL.reward import (
    compute_reward_full_xml_valid_and_iou,
    compute_reward_iou_only,
    compute_reward_valid_and_iou,
    compute_reward_valid_only,
    compute_reward_xml_iou_only,
    compute_reward_xml_valid_and_iou,
    compute_reward_xml_valid_only,
)


def _resolve_normalize(arg: bool) -> bool:
    """Resolve normalize_penalties from arg or env var."""
    if arg:
        return True
    if os.environ.get("REWARD_NORMALIZE_PENALTIES", "").lower() == "true":
        return True
    return False


def _ground_truth_to_text(ground_truth) -> str:
    if isinstance(ground_truth, dict):
        return str(ground_truth.get("ground_truth", ""))
    return "" if ground_truth is None else str(ground_truth)


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    iou_weight: float = 10.0,
    valid_base: float = 1.0,
    invalid_penalty_scale: float = 10.0,
    normalize_penalties: bool = False,
) -> float:
    """Default veRL reward: graph-valid gate plus GT Open Lines IoU."""
    return compute_score_valid_and_iou(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        iou_weight=iou_weight,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
        normalize_penalties=_resolve_normalize(normalize_penalties),
    )


def compute_score_valid_only(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    valid_base: float = 1.0,
    invalid_penalty_scale: float = 10.0,
    normalize_penalties: bool = False,
) -> float:
    """veRL reward using only graph validity and optional format gate."""
    del data_source
    extra_info = extra_info or {}
    raw_prompt = extra_info.get("raw_prompt", "")

    reward, _ = compute_reward_valid_only(
        prompt=raw_prompt,
        response=solution_str,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
        normalize_penalties=_resolve_normalize(normalize_penalties),
    )
    return float(reward)


def compute_score_iou_only(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    iou_weight: float = 10.0,
) -> float:
    """veRL reward using only GT Open Lines IoU."""
    del data_source
    extra_info = extra_info or {}
    raw_prompt = extra_info.get("raw_prompt", "")
    gt_response = extra_info.get("gt_output") or _ground_truth_to_text(ground_truth)

    reward, _ = compute_reward_iou_only(
        prompt=raw_prompt,
        response=solution_str,
        gt_response=gt_response,
        iou_weight=iou_weight,
    )
    return float(reward)


def compute_score_valid_and_iou(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    iou_weight: float = 10.0,
    valid_base: float = 1.0,
    invalid_penalty_scale: float = 10.0,
    normalize_penalties: bool = False,
) -> float:
    """veRL reward using graph-valid gate plus GT Open Lines IoU."""
    del data_source
    extra_info = extra_info or {}
    raw_prompt = extra_info.get("raw_prompt", "")
    gt_response = extra_info.get("gt_output") or _ground_truth_to_text(ground_truth)

    reward, _ = compute_reward_valid_and_iou(
        prompt=raw_prompt,
        response=solution_str,
        gt_response=gt_response,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        iou_weight=iou_weight,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
        normalize_penalties=_resolve_normalize(normalize_penalties),
    )
    return float(reward)


def compute_score_xml_valid_only(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    invalid_edges_weight: float = 1.0,
    cycles_weight: float = 1.0,
    subgraphs_weight: float = 1.0,
    format_penalty_weight: float = 0.0,
    valid_base: float = 1.0,
    invalid_penalty_scale: float = 10.0,
    normalize_penalties: bool = False,
) -> float:
    """veRL XML reward using only graph validity and optional XML format gate."""
    del data_source, ground_truth
    extra_info = extra_info or {}
    raw_prompt = extra_info.get("raw_prompt", "")

    reward, _ = compute_reward_xml_valid_only(
        prompt=raw_prompt,
        response=solution_str,
        invalid_edges_weight=invalid_edges_weight,
        cycles_weight=cycles_weight,
        subgraphs_weight=subgraphs_weight,
        format_penalty_weight=format_penalty_weight,
        valid_base=valid_base,
        invalid_penalty_scale=invalid_penalty_scale,
        normalize_penalties=_resolve_normalize(normalize_penalties),
    )
    return float(reward)


def compute_score_xml_iou_only(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    iou_weight: float = 8.0,
    improvement_weight: float = 2.0,
    precision_weight: float = 1.0,
    recall_weight: float = 1.0,
    copy_penalty: float = 2.0,
    normalize_penalties: bool = False,
) -> float:
    """veRL XML reward using GT IoU plus current-config improvement shaping."""
    del data_source
    extra_info = extra_info or {}
    raw_prompt = extra_info.get("raw_prompt", "")
    gt_response = extra_info.get("gt_output") or _ground_truth_to_text(ground_truth)

    reward, _ = compute_reward_xml_iou_only(
        prompt=raw_prompt,
        response=solution_str,
        gt_response=gt_response,
        iou_weight=iou_weight,
        improvement_weight=improvement_weight,
        precision_weight=precision_weight,
        recall_weight=recall_weight,
        copy_penalty=copy_penalty,
        normalize_penalties=_resolve_normalize(normalize_penalties),
    )
    return float(reward)


def compute_score_xml_valid_and_iou(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
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
    normalize_penalties: bool = False,
) -> float:
    """veRL XML reward using graph-valid gate plus GT Open Lines IoU."""
    del data_source
    extra_info = extra_info or {}
    raw_prompt = extra_info.get("raw_prompt", "")
    gt_response = extra_info.get("gt_output") or _ground_truth_to_text(ground_truth)

    reward, _ = compute_reward_xml_valid_and_iou(
        prompt=raw_prompt,
        response=solution_str,
        gt_response=gt_response,
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
        normalize_penalties=_resolve_normalize(normalize_penalties),
    )
    return float(reward)


def compute_score_full_xml_valid_and_iou(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
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
    normalize_penalties: bool = False,
) -> float:
    """veRL full-XML reward mirroring compute_score_xml_valid_and_iou."""
    del data_source
    extra_info = extra_info or {}
    raw_prompt = extra_info.get("raw_prompt", "")
    gt_response = extra_info.get("gt_output") or _ground_truth_to_text(ground_truth)

    reward, _ = compute_reward_full_xml_valid_and_iou(
        prompt=raw_prompt,
        response=solution_str,
        gt_response=gt_response,
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
        normalize_penalties=_resolve_normalize(normalize_penalties),
    )
    return float(reward)


def compute_score_full_xml_uniform(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
) -> float:
    """Full-XML reward with all weights = 1.0 for ablation."""
    return compute_score_full_xml_valid_and_iou(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        invalid_edges_weight=1.0,
        cycles_weight=1.0,
        subgraphs_weight=1.0,
        format_penalty_weight=0.0,
        iou_weight=1.0,
        improvement_weight=1.0,
        precision_weight=1.0,
        recall_weight=1.0,
        copy_penalty=1.0,
        valid_base=1.0,
        invalid_penalty_scale=1.0,
        normalize_penalties=_resolve_normalize(False),
    )
