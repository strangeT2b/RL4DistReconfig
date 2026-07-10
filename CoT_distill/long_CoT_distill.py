# CoT 数据蒸馏（配置外置：config/*.yaml + prompts/ + few_shot/）
from __future__ import annotations

import argparse
import json
import os
import sys
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import load_project_env  # noqa: E402
from utils.metrics_utils import (  # noqa: E402
    compute_gt_match,
    parse_open_lines_full_xml,
)
from CoT_distill.prompt_policy import (  # noqa: E402
    find_model_facing_forbidden_terms,
    has_model_facing_forbidden_terms,
)
from CoT_distill.utils import (  # noqa: E402
    assemble_trace_from_message,
    build_sample_meta_data,
    build_top_level_metadata,
    bus_of,
    clamp01,
    compute_verifier_metrics,
    extract_answer_block,
    extract_question,
    extract_solution,
    extract_think,
    extract_xml_block,
    format_few_shot_examples,
    get_api_key,
    get_base_url,
    is_disk_full_error,
    load_few_shot_examples,
    load_jsonl_records,
    load_jsonl_records_optional,
    load_text_file,
    load_yaml_config,
    message_content,
    parse_few_shot_config,
    parse_model_json_object,
    resolve_path,
)

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.datagen import SelfImprovingCoTPipeline
from camel.datagen.self_improving_cot import (
    AgentTraceEvaluation,
    ProblemResult,
    RewardTraceEvaluation,
    TraceIteration,
)
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.models.stub_model import StubTokenCounter
from camel.types import ModelPlatformType
from tqdm import tqdm

DISTILL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = DISTILL_DIR / "config" / "long_cot_distill.yaml"

formatted_datetime = datetime.now().strftime("%m-%d-%H:%M:%S")

# 并行 batch 内 per-problem few-shot（线程局部，避免多线程覆盖）
_few_shot_ctx = threading.local()


class ReconfigCoTPipeline(SelfImprovingCoTPipeline):
    """配电网重构 CoT 蒸馏 pipeline：per-problem 采样 few-shot（线程安全）。"""

    REQUIRED_USER_TEMPLATES = (
        "REASONING_TEMPLATE",
        "QUALITY_EVAL_TEMPLATE",
        "IMPROVEMENT_TEMPLATE",
        "REFLEXION_TEMPLATE",
        "REFLEXION_CONTINUE_TEMPLATE",
        "POLISH_TEMPLATE",
        "GT_GUIDED_CORRECTION_TEMPLATE",
    )

    def __init__(
        self,
        *,
        few_shot_pool: List[dict],
        few_shot_num_samples: int,
        few_shot_seed: int,
        use_reflexion: bool = False,
        reflexion_agent: Optional[ChatAgent] = None,
        gt_correction_cfg: Optional[dict] = None,
        quality_cfg: Optional[dict] = None,
        timeout_cfg: Optional[dict] = None,
        user_templates: Optional[dict] = None,
        **kwargs: Any,
    ):
        # 父类 few_shot_examples 仅作占位；实际由本类 per-problem 注入
        super().__init__(few_shot_examples=None, **kwargs)
        self.few_shot_pool = few_shot_pool
        self.few_shot_num_samples = max(1, few_shot_num_samples)
        self.few_shot_seed = few_shot_seed
        self.use_reflexion = bool(use_reflexion)
        self.reflexion_agent = reflexion_agent
        self._gt_correction_cfg = parse_gt_correction_cfg(gt_correction_cfg or {})
        self._quality_cfg = parse_quality_cfg(quality_cfg or {})
        self._timeout_cfg = parse_timeout_cfg(timeout_cfg or {})
        for name, template in (user_templates or {}).items():
            if template:
                setattr(self, name, template)
        missing_templates = [
            name
            for name in self.REQUIRED_USER_TEMPLATES
            if not str(getattr(self, name, "")).strip()
        ]
        if missing_templates:
            raise ValueError(
                "缺少 user template 配置或文件为空: "
                + ", ".join(missing_templates)
            )

    def _sample_few_shot_str(self, problem: dict) -> str:
        problem_text = str(problem.get("problem", "")).strip()
        # 跨 bus 系统 demo：排除与测试题同 bus 的 demo，避免模型照抄 demo 答案。
        # 同 bus demo 的完整答案会被模型走捷径抄走，不推理；只用异 bus demo 教风格。
        pbus = bus_of(problem_text)
        pool = self.few_shot_pool
        if pbus is not None:
            cross = [d for d in pool if bus_of(str(d.get("problem", ""))) != pbus]
            if len(cross) >= self.few_shot_num_samples:
                pool = cross  # 异 bus demo 够抽 k 条，就用它；否则退回全池
        k = min(self.few_shot_num_samples, len(pool))
        if k <= 0:
            return ""
        rng = random.Random(f"{self.few_shot_seed}:{problem_text}:{k}")
        picked = list(pool) if k >= len(pool) else rng.sample(pool, k)
        return format_few_shot_examples(picked)

    def process_problem(
        self, problem: Dict, rationalization: bool = False
    ):
        _few_shot_ctx.examples_str = self._sample_few_shot_str(problem)
        _few_shot_ctx.solution = problem.get("solution")
        return self._process_problem_with_optional_gt_correction(
            problem,
            rationalization=rationalization,
        )

    def _active_few_shot_block(self) -> str:
        examples_str = getattr(_few_shot_ctx, "examples_str", "") or ""
        if not examples_str:
            return ""
        return f"Examples:\n{examples_str}"

    @staticmethod
    def _assemble_trace(msg) -> str:
        return assemble_trace_from_message(msg)

    def generate_reasoning_trace(self, problem: str) -> str:
        max_retries = self._timeout_cfg["camel_retry_max_retries"]
        delay = self._timeout_cfg["camel_retry_delay"]
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                self.reason_agent.reset()
                few_shot_block = self._active_few_shot_block()
                prompt = self.REASONING_TEMPLATE.format(
                    problem=problem,
                    few_shot_examples=few_shot_block,
                )
                response = self.reason_agent.step(prompt)
                return self._assemble_trace(response.msg)
            except Exception as e:
                last_exc = e
                if attempt >= max_retries:
                    raise
                print(
                    f"[WARN] generate_reasoning_trace 失败 ({attempt + 1}/{max_retries + 1}): {e} "
                    f"— {delay:.1f}s 后重试..."
                )
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
        if last_exc is not None:
            raise last_exc
        return ""

    def _candidate_rank(self, eval_result: dict) -> tuple:
        metrics = eval_result["verifier_metrics"]
        features = eval_result["decision_features"]
        return (
            int(metrics["format"] == 1.0),
            int(metrics["graph_validity"] == 1.0),
            int(metrics["gt_exact_match"] == 1.0),
            int(metrics["copied_input_open"] == 0.0),
            int(metrics.get("hit_count_hidden", 0)),
            float(metrics.get("iou_hidden", 0.0)),
            float(metrics.get("recall_hidden", 0.0)),
            float(metrics.get("precision_hidden", 0.0)),
            float(features.get("reasoning_quality", 0.0)),
        )

    def generate_reasoning_trace_rejection(self, problem: str) -> str:
        few_shot_block = self._active_few_shot_block()
        prompt = self.REASONING_TEMPLATE.format(
            problem=problem,
            few_shot_examples=few_shot_block,
        )
        candidate_traces: List[str] = []
        if "n" in self.reason_agent.model_backend.model_config_dict:
            self.reason_agent.model_backend.model_config_dict["n"] = (
                self.rejection_sampling_n
            )
            responses = self.reason_agent.step(prompt)
            candidate_traces = [self._assemble_trace(choice) for choice in responses.msgs]
        else:
            sampling_n = (
                self.rejection_sampling_n
                if self.rejection_sampling_n is not None
                else 1
            )
            for _i in range(sampling_n):
                candidate_traces.append(self.generate_reasoning_trace(problem))

        best_trace = None
        best_rank = None
        for trace in candidate_traces:
            eval_result = self.evaluate_trace_structured(problem, trace)
            rank = self._candidate_rank(eval_result)
            if best_rank is None or rank > best_rank:
                best_trace = trace
                best_rank = rank
        return best_trace or ""

    @staticmethod
    def _extract_xml_block(text: str, tag: str) -> str:
        return extract_xml_block(text, tag)

    @staticmethod
    def _extract_think(trace: str) -> str:
        return extract_think(trace)

    @staticmethod
    def _extract_answer_block(trace: str) -> str:
        return extract_answer_block(trace)

    @classmethod
    def _build_eval_trace_view(cls, trace: str) -> str:
        """Build an eval-only view that avoids model-special <think> tags.

        The SFT trace kept by the pipeline remains unchanged as
        <think>...</think><answer>...</answer>. Only the eval agent sees this
        remapped view, so the verifier and downstream SFT format keep using the
        original trace.
        """
        reasoning = cls._extract_think(trace)
        answer = cls._extract_xml_block(trace, "answer")
        if not reasoning and not answer:
            return trace or ""
        parts = [
            "<reasoning_trace>",
            reasoning.strip() if reasoning else "",
            "</reasoning_trace>",
        ]
        if answer:
            parts.extend(
                [
                    "<final_answer_block>",
                    answer.strip(),
                    "</final_answer_block>",
                ]
            )
        return "\n".join(parts)

    def _compute_verifier_metrics(self, problem: str, trace: str) -> dict:
        gt_solution = getattr(_few_shot_ctx, "solution", "") or ""
        return compute_verifier_metrics(
            problem,
            trace,
            gt_solution,
            min_iou_accept=self._quality_cfg["min_iou_accept"],
        )

    def _build_verifier_summary(self, metrics: dict) -> str:
        validity = "valid" if metrics["graph_validity"] == 1.0 else "invalid"
        if metrics["format"] == 0.0:
            validity = "XML parse failed"
        if metrics["iou_hidden"] >= self._quality_cfg["min_iou_accept"]:
            similarity = "high"
        elif metrics["graph_validity"] == 1.0:
            similarity = "needs improvement"
        else:
            similarity = "not assessed until topology is valid"
        return (
            f"format_ok={bool(metrics['format'])}; graph={validity}; "
            f"invalid_edges={metrics['invalid_edges']:.0f}; cycles={metrics['cycles']:.0f}; "
            f"subgraphs={metrics['subgraphs']:.0f}; topology_similarity={similarity}; "
            f"copied_current_open_lines={bool(metrics['copied_input_open'])}."
        )

    def _sanitize_model_facing_feedback(self, text: str) -> str:
        if not text:
            return ""
        pieces = re.split(r"(?<=[.!?])\s+", str(text).strip())
        clean: List[str] = []
        for piece in pieces:
            if find_model_facing_forbidden_terms(piece):
                continue
            if piece.strip():
                clean.append(piece.strip())
        return " ".join(clean).strip()

    def _build_model_facing_feedback(self, metrics: dict, eval_feedback: str) -> str:
        parts: List[str] = []
        if metrics["format"] == 0.0:
            parts.append(
                "The answer XML did not parse as full XML. Produce exactly one <answer> block "
                "with <open_lines>, <node_voltages>, and <system_loss>."
            )
        if metrics["invalid_edges"] > 0:
            parts.append("Some proposed open lines are not in the input Lines list.")
        if metrics["cycles"] > 0:
            parts.append("The closed graph still contains cycles; add or move sectionalizing breaks.")
        if metrics["subgraphs"] > 0:
            parts.append("The closed graph is disconnected; preserve a source-side path to every branch.")
        if (
            metrics["format"]
            and metrics["graph_validity"]
            and metrics["iou_hidden"] < self._quality_cfg["min_iou_accept"]
        ):
            parts.append(
                "The topology is valid, but the selected branch exchanges still miss important "
                "switching choices. Re-check heavy-load paths, weak-voltage tails, and tie-loop cuts."
            )
        if metrics["copied_input_open"] == 1.0:
            parts.append("The answer appears to copy the current open-line set; propose a real reconfiguration.")
        clean_eval_feedback = ""
        if not str(eval_feedback).startswith("eval_agent_failed:"):
            clean_eval_feedback = self._sanitize_model_facing_feedback(eval_feedback)
        if clean_eval_feedback:
            parts.append(f"Reasoning-quality feedback: {clean_eval_feedback}")
        feedback = " ".join(parts) or "Improve the reasoning specificity and ensure the final answer matches the reasoning."
        return self._sanitize_model_facing_feedback(feedback) or "Improve the reasoning specificity and topology checks."

    def _rubric_reasoning_quality(self, rubric: dict) -> float:
        return (
            clamp01(rubric.get("logic_relevance"))
            + clamp01(rubric.get("uses_problem_data"))
            + clamp01(rubric.get("reasoning_coherence"))
            + clamp01(rubric.get("key_factors"))
        ) / 4.0

    def _build_pipeline_scores(self, metrics: dict, rubric: dict, feedback: str) -> dict:
        logic_score = clamp01(rubric.get("logic_relevance"))
        data_score = clamp01(rubric.get("uses_problem_data"))
        coherence_score = clamp01(rubric.get("reasoning_coherence"))
        key_factor_score = clamp01(rubric.get("key_factors"))
        no_leak_score = clamp01(rubric.get("no_gt_leakage"))
        length_score = clamp01(rubric.get("length_control"))
        return {
            "correctness": min(
                metrics["format"],
                metrics["graph_validity"],
                metrics["answer_alignment"],
                no_leak_score,
            ),
            "clarity": (coherence_score + length_score) / 2.0,
            "completeness": (logic_score + data_score + key_factor_score) / 3.0,
            "feedback": feedback,
        }

    def evaluate_trace_structured(self, problem: str, trace: str, solution=None) -> dict:
        metrics = self._compute_verifier_metrics(problem, trace)
        local_summary = self._build_verifier_summary(metrics)

        rubric: dict[str, Any] = {
            "logic_relevance": 0.0,
            "uses_problem_data": 0.0,
            "reasoning_coherence": 0.0,
            "key_factors": 0.0,
            "no_gt_leakage": 0.0,
            "length_control": 0.0,
            "feedback": "",
        }
        eval_feedback = ""
        try:
            self.evaluate_agent.reset()
            eval_trace = self._build_eval_trace_view(trace)
            prompt = self.QUALITY_EVAL_TEMPLATE.format(
                problem=problem,
                trace=eval_trace,
                local_summary=local_summary,
            )
            response = self.evaluate_agent.step(prompt)
            raw = message_content(response.msg)
            parsed = parse_model_json_object(raw)
            eval_feedback = str(parsed.get("feedback", "")).strip()
            for key in (
                "logic_relevance",
                "uses_problem_data",
                "reasoning_coherence",
                "key_factors",
                "no_gt_leakage",
                "length_control",
            ):
                rubric[key] = clamp01(parsed.get(key, 0.0))
        except Exception as e:
            eval_feedback = f"eval_agent_failed: {e}"

        rubric["feedback"] = (
            ""
            if str(eval_feedback).startswith("eval_agent_failed:")
            else self._sanitize_model_facing_feedback(eval_feedback)
        )
        feedback = self._build_model_facing_feedback(metrics, eval_feedback)
        pipeline_scores = self._build_pipeline_scores(metrics, rubric, feedback)
        pipeline_passes = self._check_score_threshold(
            {k: v for k, v in pipeline_scores.items() if k != "feedback"}
        )
        reasoning_quality = self._rubric_reasoning_quality(rubric)
        bridgeable = self._is_gt_correction_bridgeable(problem, metrics, solution or "")
        decision_features = {
            "is_parseable": metrics["format"] == 1.0,
            "is_graph_valid": metrics["graph_validity"] == 1.0,
            "is_exact": metrics["gt_exact_match"] == 1.0,
            "is_valid_not_exact": (
                metrics["graph_validity"] == 1.0 and metrics["gt_exact_match"] < 1.0
            ),
            "is_copied_input": metrics["copied_input_open"] == 1.0,
            "reasoning_quality": float(reasoning_quality),
            "no_forbidden_leakage": (
                clamp01(rubric.get("no_gt_leakage"))
                >= self._quality_cfg["min_no_leak_accept"]
            ),
            "bridgeable": bool(bridgeable),
            "hit_count": int(metrics.get("hit_count_hidden", 0)),
            "pipeline_score_passes": bool(pipeline_passes),
        }
        return {
            "verifier_metrics": metrics,
            "rubric": rubric,
            "pipeline_scores": pipeline_scores,
            "decision_features": decision_features,
        }

    def evaluate_trace(self, problem: str, trace: str, solution=None):
        return self.evaluate_trace_structured(problem, trace, solution)["pipeline_scores"]

    def _has_gt_correction_leakage(self, trace: str) -> bool:
        return has_model_facing_forbidden_terms(trace)

    def _is_gt_correction_bridgeable(self, problem: str, metrics: dict, solution: str) -> bool:
        cfg = self._gt_correction_cfg
        if not cfg["enabled"] or not solution:
            return False
        if cfg["require_valid"] and metrics["graph_validity"] < 1.0:
            return False
        if metrics["gt_exact_match"] >= 1.0:
            return False
        bus = bus_of(problem)
        min_hit_by_bus = cfg.get("min_hit_by_bus") or {}
        min_hit = min_hit_by_bus.get(bus) if bus is not None else None
        if min_hit is not None and metrics.get("hit_count_hidden", 0) >= min_hit:
            return True
        return (
            metrics["iou_hidden"] >= cfg["min_iou"]
            or metrics["recall_hidden"] >= cfg["min_recall"]
        )

    def _decide_improvement_mode(
        self,
        eval_result: dict,
        *,
        iteration: int,
        max_iterations: int,
        problem: str,
        solution: str,
    ) -> dict:
        metrics = eval_result["verifier_metrics"]
        features = eval_result["decision_features"]
        quality = float(features.get("reasoning_quality", 0.0))
        parse_ok = bool(features.get("is_parseable"))
        valid = bool(features.get("is_graph_valid"))
        exact = bool(features.get("is_exact"))
        no_leak = bool(features.get("no_forbidden_leakage"))
        accept_q = self._quality_cfg["min_reasoning_quality_accept"]
        polish_q = self._quality_cfg["min_reasoning_quality_polish"]

        if parse_ok and valid and exact and no_leak and quality >= accept_q:
            return {
                "mode": "accept",
                "reason": f"exact valid trace with sufficient reasoning quality ({quality:.3f})",
            }
        if parse_ok and valid and exact:
            return {
                "mode": "polish" if iteration < max_iterations else "record_unusable",
                "reason": f"answer is exact but reasoning quality needs polish ({quality:.3f} < {polish_q:.3f})",
                "lock_answer": True,
            }
        if self._is_gt_correction_bridgeable(problem, metrics, solution):
            return {
                "mode": "gt_guided_correction" if iteration < max_iterations else "record_unusable",
                "reason": (
                    "valid close miss selected for guided correction "
                    f"(hit={metrics.get('hit_count_hidden', 0)}, "
                    f"target={metrics.get('target_open_count', 0)})"
                ),
            }
        if iteration < max_iterations and self.use_reflexion and self.reflexion_agent is not None:
            return {
                "mode": "reflexion",
                "reason": "candidate is not acceptable and improvement iterations remain",
            }
        return {
            "mode": "record_unusable",
            "reason": "no safe improvement mode remains or maximum iterations reached",
        }

    def _force_solution_answer(self, trace: str, solution: str) -> str:
        answer = self._extract_answer_block(solution) or solution.strip()
        think = self._extract_think(trace)
        if not think:
            think = (trace or "").strip()
        return "<think>\n" + think.strip() + "\n</think>\n" + answer.strip()

    def gt_guided_correct_trace(self, problem: str, trace: str, solution: str) -> str:
        if not solution:
            return trace
        solution_answer = self._extract_answer_block(solution) or solution.strip()
        prompt = self.GT_GUIDED_CORRECTION_TEMPLATE.format(
            problem=problem,
            trace=trace,
            solution=solution_answer,
        )
        corrected = self._call_reason_agent_for_improvement(prompt)
        if not corrected:
            return trace
        corrected = self._force_solution_answer(corrected, solution)
        if self._has_gt_correction_leakage(corrected):
            print("[WARN] GT-guided correction leaked forbidden wording; using forced-answer trace.")
            return self._force_solution_answer(trace, solution)
        return corrected

    def _append_iteration_history(
        self,
        history: list,
        *,
        iteration: int,
        trace: str,
        eval_dict: dict,
    ) -> None:
        scores = {k: v for k, v in eval_dict.items() if k != "feedback"}
        if self.evaluator:
            history.append(
                TraceIteration(
                    iteration=iteration,
                    trace=trace,
                    evaluation=RewardTraceEvaluation(**eval_dict),
                )
            )
        else:
            history.append(
                TraceIteration(
                    iteration=iteration,
                    trace=trace,
                    evaluation=AgentTraceEvaluation(
                        **scores,
                        feedback=eval_dict["feedback"],
                    ),
                )
            )

    def _process_problem_with_optional_gt_correction(
        self,
        problem: Dict,
        rationalization: bool = False,
    ) -> ProblemResult:
        self.validate_problem_format(problem)

        problem_text = problem["problem"]
        solution_text = problem.get("solution", "")
        if self.rejection_sampling_n:
            raw_initial = self.generate_reasoning_trace_rejection(problem_text)
            initial_origin = "rejection_sampling"
        else:
            raw_initial = self.generate_reasoning_trace(problem_text)
            initial_origin = "initial"

        working_sft_trace = raw_initial
        final_sft_trace = ""
        final_origin = ""
        status = "unusable_not_evaluated"
        usable_for_sft = False
        improvement_history = []
        raw_model_outputs = [
            {
                "iteration": 0,
                "origin": initial_origin,
                "trace": raw_initial,
            }
        ]
        trajectory_audit = []
        scores = {}
        eval_dict = {}
        final_eval_result: dict = {}

        for iteration in range(self.max_iterations + 1):
            eval_result = self.evaluate_trace_structured(
                problem_text,
                working_sft_trace,
                solution_text,
            )
            final_eval_result = eval_result
            eval_dict = eval_result["pipeline_scores"]
            scores = {k: v for k, v in eval_dict.items() if k != "feedback"}
            self._append_iteration_history(
                improvement_history,
                iteration=iteration,
                trace=working_sft_trace,
                eval_dict=eval_dict,
            )
            decision = self._decide_improvement_mode(
                eval_result,
                iteration=iteration,
                max_iterations=self.max_iterations,
                problem=problem_text,
                solution=solution_text,
            )
            trajectory_audit.append(
                {
                    "iteration": iteration,
                    "origin": (
                        raw_model_outputs[-1]["origin"]
                        if raw_model_outputs
                        else "unknown"
                    ),
                    "working_sft_trace": working_sft_trace,
                    "eval_result": eval_result,
                    "decision": decision,
                }
            )

            mode = decision["mode"]
            if mode == "accept":
                final_sft_trace = working_sft_trace
                final_origin = trajectory_audit[-1]["origin"]
                status = "accepted"
                usable_for_sft = True
                break
            if mode == "record_unusable":
                status = (
                    "unusable_max_iterations"
                    if iteration >= self.max_iterations
                    else "unusable_low_quality"
                )
                break
            if iteration >= self.max_iterations:
                status = "unusable_max_iterations"
                break

            try:
                improved_trace = self.improve_trace(
                    problem_text,
                    working_sft_trace,
                    eval_dict.get("feedback", ""),
                    solution_text,
                    mode=mode,
                    eval_result=eval_result,
                )
            except Exception as e:
                print(f"[WARN] {mode} improvement failed; recording unusable trace: {e}")
                status = f"unusable_{mode}_failed"
                break
            if not improved_trace:
                status = f"unusable_{mode}_empty"
                break
            working_sft_trace = improved_trace
            raw_model_outputs.append(
                {
                    "iteration": iteration + 1,
                    "origin": mode,
                    "trace": improved_trace,
                    "parent_iteration": iteration,
                }
            )

        current_trace = final_sft_trace or working_sft_trace
        boxed_answer_success = self._check_boxed_answers(
            problem.get("solution", ""), current_trace
        )

        result = ProblemResult(
            id=problem.get("id", ""),
            type=problem.get("type", ""),
            problem=problem_text,
            solution=problem.get("solution", ""),
            final_trace=current_trace,
            agent_evaluate_success=self._check_score_threshold(scores)
            if scores
            else None,
            boxed_answer_success=boxed_answer_success,
            improvement_history=improvement_history,
        )

        if self.output_path:
            with self.lock:
                try:
                    with open(self.output_path, "r") as f:
                        data = json.load(f)

                    cleaned_result = self.clean_json(result.model_dump())
                    meta_data = dict(problem.get("meta_data") or {})
                    cleaned_result.update(
                        self.clean_json(
                            {
                                "pipeline_version": "long_cot_v3_explicit_loop",
                                "status": status,
                                "usable_for_sft": usable_for_sft,
                                "working_sft_trace": working_sft_trace,
                                "final_sft_trace": final_sft_trace,
                                "final_eval_result": final_eval_result,
                                "raw_model_outputs": raw_model_outputs,
                                "trajectory_audit": trajectory_audit,
                                "final_origin": final_origin,
                                "meta_data": meta_data,
                                **build_top_level_metadata(meta_data),
                            }
                        )
                    )
                    data["traces"].append(cleaned_result)
                    self.safe_write_json(self.output_path, data)
                except Exception as e:
                    print(f"[ERROR] Error writing result to file: {e}")

        return result

    def _call_reason_agent_for_improvement(self, prompt: str) -> str:
        self.reason_agent.reset()
        OPEN = chr(60) + "think" + chr(62)
        max_retry = self._timeout_cfg.get("improve_retry", 5)
        assembled = ""
        for _attempt in range(max_retry):
            response = self.reason_agent.step(prompt)
            assembled = self._assemble_trace(response.msg)
            if OPEN in assembled:  # 产出了 think 推理块
                return assembled
            self.reason_agent.reset()  # 空 reasoning，重采
        return assembled  # 重试用尽仍空，返回最后一条（至少有 answer）

    def improve_trace_rewrite(self, problem, trace, feedback, solution=None):
        prompt = self.IMPROVEMENT_TEMPLATE.format(
            problem=problem,
            trace=trace,
            feedback=self._sanitize_model_facing_feedback(feedback),
        )
        return self._call_reason_agent_for_improvement(prompt)

    @classmethod
    def _assemble_reflexion_trace(
        cls,
        *,
        old_think: str,
        reflection: str,
        continuation_trace: str,
    ) -> str:
        continuation_think = cls._extract_think(continuation_trace)
        final_answer = cls._extract_answer_block(continuation_trace)
        if not final_answer:
            return ""
        think_parts = [
            part.strip()
            for part in (old_think, reflection, continuation_think)
            if part and part.strip()
        ]
        return (
            "<think>\n"
            + "\n\n".join(think_parts)
            + "\n</think>\n"
            + final_answer
        )

    def generate_reflexion(self, problem: str, old_think: str, feedback: str) -> str:
        if self.reflexion_agent is None:
            return ""
        self.reflexion_agent.reset()
        prompt = self.REFLEXION_TEMPLATE.format(
            problem=problem,
            old_think=old_think,
            feedback=self._sanitize_model_facing_feedback(feedback),
        )
        response = self.reflexion_agent.step(prompt)
        text = message_content(response.msg)
        # Keep only the self-reflection paragraph. The system prompt forbids
        # leaking meta terms, but this guard catches accidental wrappers.
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE).strip()
        return self._sanitize_model_facing_feedback(text)

    def improve_trace_with_reflexion(self, problem, trace, feedback, solution=None) -> str:
        old_think = self._extract_think(trace)
        if not old_think:
            return self.improve_trace_rewrite(problem, trace, feedback, solution=solution)

        reflection = self.generate_reflexion(problem, old_think, feedback).strip()
        if not reflection:
            return self.improve_trace_rewrite(problem, trace, feedback, solution=solution)

        prompt = self.REFLEXION_CONTINUE_TEMPLATE.format(
            problem=problem,
            old_think=old_think,
            reflection=reflection,
        )
        continuation_trace = self._call_reason_agent_for_improvement(prompt)
        assembled = self._assemble_reflexion_trace(
            old_think=old_think,
            reflection=reflection,
            continuation_trace=continuation_trace,
        )
        return assembled or self.improve_trace_rewrite(
            problem,
            trace,
            feedback,
            solution=solution,
        )

    def improve_trace_polish(self, problem, trace, feedback, eval_result=None) -> str:
        answer = self._extract_answer_block(trace)
        if not answer:
            return trace
        old_open = parse_open_lines_full_xml(trace)
        prompt = self.POLISH_TEMPLATE.format(
            problem=problem,
            trace=trace,
            feedback=self._sanitize_model_facing_feedback(feedback),
        )
        polished = self._call_reason_agent_for_improvement(prompt)
        if not polished:
            return trace
        think = self._extract_think(polished) or self._extract_think(trace)
        candidate = "<think>\n" + think.strip() + "\n</think>\n" + answer.strip()
        new_open = parse_open_lines_full_xml(candidate)
        if old_open and compute_gt_match(new_open, old_open)[0] != 1.0:
            return trace
        return candidate

    def improve_trace(
        self,
        problem,
        trace,
        feedback,
        solution=None,
        *,
        mode=None,
        eval_result=None,
    ):
        mode = mode or (
            "reflexion"
            if self.use_reflexion and self.reflexion_agent is not None
            else "polish"
        )
        if mode == "reflexion":
            try:
                return self.improve_trace_with_reflexion(
                    problem,
                    trace,
                    feedback,
                    solution=solution,
                )
            except Exception as e:
                print(f"[WARN] reflexion repair failed, fallback to rewrite repair: {e}")
                return self.improve_trace_rewrite(problem, trace, feedback, solution=solution)
        if mode == "polish":
            return self.improve_trace_polish(
                problem,
                trace,
                feedback,
                eval_result=eval_result,
            )
        if mode == "gt_guided_correction":
            return self.gt_guided_correct_trace(problem, trace, solution or "")
        return self.improve_trace_rewrite(problem, trace, feedback, solution=solution)


def parse_timeout_cfg(pipeline_cfg: dict) -> dict:
    """解析 API / Agent / 样本级超时与重试配置。"""
    cfg = pipeline_cfg or {}
    return {
        "request_timeout": float(cfg.get("request_timeout", 600)),
        "max_retries": int(cfg.get("max_retries", 5)),
        "agent_retry_attempts": int(cfg.get("agent_retry_attempts", 5)),
        "agent_retry_delay": float(cfg.get("agent_retry_delay", 2.0)),
        "agent_step_timeout": float(cfg.get("agent_step_timeout", 600)),
        "camel_retry_max_retries": int(cfg.get("camel_retry_max_retries", 5)),
        "camel_retry_delay": float(cfg.get("camel_retry_delay", 2.0)),
        "problem_max_retries": int(cfg.get("problem_max_retries", 3)),
        "problem_retry_delay": float(cfg.get("problem_retry_delay", 5.0)),
        "improve_retry": int(cfg.get("improve_retry", 5)),
    }


def parse_quality_cfg(pipeline_cfg: dict) -> dict:
    """Quality/filter thresholds for long-CoT distillation."""
    cfg = pipeline_cfg or {}
    return {
        "min_iou_accept": float(cfg.get("min_iou_accept", 0.75)),
        "min_reasoning_quality_accept": float(
            cfg.get("min_reasoning_quality_accept", 0.70)
        ),
        "min_reasoning_quality_polish": float(
            cfg.get("min_reasoning_quality_polish", 0.70)
        ),
        "min_no_leak_accept": float(cfg.get("min_no_leak_accept", 0.99)),
    }


def parse_gt_correction_cfg(pipeline_cfg: dict) -> dict:
    cfg = pipeline_cfg or {}
    raw = cfg.get("gt_correction") or {}
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    if not isinstance(raw, dict):
        raw = {}
    raw_hit = raw.get("min_hit_by_bus") or {}
    min_hit_by_bus = {}
    if isinstance(raw_hit, dict):
        for key, value in raw_hit.items():
            try:
                min_hit_by_bus[int(key)] = int(value)
            except (TypeError, ValueError):
                continue
    if not min_hit_by_bus:
        min_hit_by_bus = {33: 4, 69: 4, 84: 10}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "min_iou": float(raw.get("min_iou", 0.5)),
        "min_recall": float(raw.get("min_recall", 0.6)),
        "require_valid": bool(raw.get("require_valid", True)),
        "min_hit_by_bus": min_hit_by_bus,
    }


def run_pipeline_with_progress(
    pipeline,
    problems,
    rationalization=False,
    *,
    timeout_cfg: Optional[dict] = None,
):
    timeout_cfg = parse_timeout_cfg(timeout_cfg or {})
    problem_max_retries = timeout_cfg["problem_max_retries"]
    problem_retry_delay = timeout_cfg["problem_retry_delay"]

    def process_one(problem: dict):
        delay = problem_retry_delay
        last_exc: Optional[Exception] = None
        for attempt in range(problem_max_retries + 1):
            try:
                return pipeline.process_problem(
                    problem=problem,
                    rationalization=rationalization,
                )
            except Exception as e:
                last_exc = e
                if is_disk_full_error(e):
                    raise SystemExit(
                        "[FATAL] 检测到磁盘空间/配额不足导致写入失败，程序立即停止。"
                    ) from e
                if attempt >= problem_max_retries:
                    raise
                print(
                    f"[WARN] 样本处理失败 ({attempt + 1}/{problem_max_retries + 1}): {e} "
                    f"— {delay:.1f}s 后重试..."
                )
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
        if last_exc is not None:
            raise last_exc

    results = []
    total_problems = len(problems)
    processed = 0

    with tqdm(total=total_problems, desc="CoT distill", unit="sample") as pbar:
        while processed < total_problems:
            batch_size = pipeline.batch_processor.batch_size
            batch = problems[processed : processed + batch_size]
            batch_start_time = time.time()
            batch_success = True

            with ThreadPoolExecutor(max_workers=pipeline.batch_processor.max_workers) as executor:
                futures = [
                    executor.submit(process_one, problem)
                    for problem in batch
                ]
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        if is_disk_full_error(e):
                            raise SystemExit(
                                "[FATAL] 检测到磁盘空间/配额不足导致写入失败，程序立即停止。"
                            ) from e
                        print(f"[WARN] 样本最终失败（已重试 {problem_max_retries} 次）: {e}")
                        batch_success = False
                    finally:
                        pbar.update(1)

            processed += len(batch)
            pipeline.batch_processor.adjust_batch_size(
                batch_success, time.time() - batch_start_time
            )
    return results


def load_user_templates(prompts_cfg: dict, *, config_dir: Path) -> dict:
    template_keys = {
        "reasoning_template": "REASONING_TEMPLATE",
        "quality_eval_template": "QUALITY_EVAL_TEMPLATE",
        "improvement_template": "IMPROVEMENT_TEMPLATE",
        "reflexion_template": "REFLEXION_TEMPLATE",
        "reflexion_continue_template": "REFLEXION_CONTINUE_TEMPLATE",
        "polish_template": "POLISH_TEMPLATE",
        "gt_guided_correction_template": "GT_GUIDED_CORRECTION_TEMPLATE",
    }
    templates = {}
    for cfg_key, attr_name in template_keys.items():
        raw = (prompts_cfg or {}).get(cfg_key)
        if raw:
            templates[attr_name] = load_text_file(
                resolve_path(str(raw), base_dir=config_dir)
            )
    return templates


def openai_compatible_client_kwargs() -> dict[str, Any]:
    # api.apigzt.xyz currently blocks the OpenAI Python client's default UA.
    # A curl-like UA keeps the request compatible with that gateway.
    return {"default_headers": {"User-Agent": "curl/8.7.1"}}


def merge_config_with_args(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI 显式传入的非 None 值覆盖 YAML。"""
    out = json.loads(json.dumps(cfg))  # deep copy

    if args.cot_history_path is not None:
        out.setdefault("data", {})["cot_history_path"] = args.cot_history_path
    if args.qa_data_path is not None:
        out.setdefault("data", {})["qa_data_path"] = args.qa_data_path
    if args.output_dir is not None:
        out.setdefault("data", {})["output_dir"] = args.output_dir
    if args.reason_model_name is not None:
        out.setdefault("models", {}).setdefault("reason", {})["name"] = args.reason_model_name
    if args.reason_base_url is not None:
        out.setdefault("models", {}).setdefault("reason", {})["base_url"] = args.reason_base_url
    if args.eval_model_name is not None:
        out.setdefault("models", {}).setdefault("eval", {})["name"] = args.eval_model_name
    if args.eval_base_url is not None:
        out.setdefault("models", {}).setdefault("eval", {})["base_url"] = args.eval_base_url
    if args.preserve_metadata_keys is not None:
        keys = [k.strip() for k in args.preserve_metadata_keys.split(",") if k.strip()]
        out["preserve_metadata_keys"] = keys
    if args.few_shot_num_samples is not None or args.few_shot_seed is not None:
        fs = out.get("few_shot")
        if not isinstance(fs, dict):
            raise ValueError("配置 few_shot 需为 {path, num_samples, seed} 对象")
        if args.few_shot_num_samples is not None:
            fs["num_samples"] = args.few_shot_num_samples
        if args.few_shot_seed is not None:
            fs["seed"] = args.few_shot_seed
    if args.batch_size is not None:
        out.setdefault("pipeline", {})["batch_size"] = args.batch_size
    if args.max_workers is not None:
        out.setdefault("pipeline", {})["max_workers"] = args.max_workers
    if args.use_reflexion is not None:
        out.setdefault("pipeline", {})["use_reflexion"] = args.use_reflexion
    gt_cfg = out.setdefault("pipeline", {}).setdefault("gt_correction", {})
    if args.use_gt_correction is not None:
        gt_cfg["enabled"] = args.use_gt_correction
    if args.gt_correction_min_iou is not None:
        gt_cfg["min_iou"] = args.gt_correction_min_iou
    if args.gt_correction_min_recall is not None:
        gt_cfg["min_recall"] = args.gt_correction_min_recall
    if args.gt_correction_require_valid is not None:
        gt_cfg["require_valid"] = args.gt_correction_require_valid
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="CoT distillation runner（YAML 配置 + 任务可切换）")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML 配置文件路径（默认 CoT_distill/config/long_cot_distill.yaml）",
    )
    parser.add_argument("--cot_history_path", type=str, default=None, help="覆盖 data.cot_history_path")
    parser.add_argument("--qa_data_path", type=str, default=None, help="覆盖 data.qa_data_path")
    parser.add_argument("--qa_start", type=int, default=0, help="切片起始索引（含）")
    parser.add_argument("--qa_end", type=int, default=-1, help="切片结束索引（不含），-1 表示到末尾")
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="指定行索引（逗号分隔，如 '2619,456,3928'），优先于 qa_start/qa_end；可让一批不连续样本在一个 batch 内并发跑",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="覆盖 data.output_dir")
    parser.add_argument("--reason_model_name", type=str, default=None)
    parser.add_argument("--reason_base_url", type=str, default=None)
    parser.add_argument("--eval_model_name", type=str, default=None)
    parser.add_argument("--eval_base_url", type=str, default=None)
    parser.add_argument(
        "--preserve_metadata_keys",
        type=str,
        default=None,
        help="覆盖 preserve_metadata_keys，逗号分隔；建议包含 meta",
    )
    parser.add_argument(
        "--few-shot-num-samples",
        type=int,
        default=None,
        dest="few_shot_num_samples",
        help="每条蒸馏样本随机采样的 few-shot 条数（默认 2）",
    )
    parser.add_argument(
        "--few-shot-seed",
        type=int,
        default=None,
        dest="few_shot_seed",
        help="few-shot 采样基础种子（与 problem 文本组合保证可复现）",
    )
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--max-workers", type=int, default=None, dest="max_workers")
    parser.add_argument(
        "--use-reflexion",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="use_reflexion",
        help="启用 reflexion continuation repair；默认读取 YAML",
    )
    parser.add_argument(
        "--use-gt-correction",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="use_gt_correction",
        help="启用 GT-guided final correction；默认读取 YAML",
    )
    parser.add_argument(
        "--gt-correction-min-iou",
        type=float,
        default=None,
        help="候选 trace hidden IoU 达到该阈值时才做 GT correction",
    )
    parser.add_argument(
        "--gt-correction-min-recall",
        type=float,
        default=None,
        help="候选 trace hidden recall 达到该阈值时才做 GT correction",
    )
    parser.add_argument(
        "--gt-correction-require-valid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="GT correction 前是否要求候选拓扑 valid；默认读取 YAML",
    )
    return parser.parse_args()


def build_agents(cfg: dict, config_dir: Path, *, timeout_cfg: Optional[dict] = None):
    agents_cfg = cfg.get("agents") or {}
    prompts_cfg = cfg.get("prompts") or {}
    models_cfg = cfg.get("models") or {}
    pipeline_cfg = cfg.get("pipeline") or {}
    tcfg = parse_timeout_cfg(timeout_cfg or {})

    reason_prompt = load_text_file(resolve_path(prompts_cfg["reason_system"], base_dir=config_dir))
    eval_prompt = load_text_file(resolve_path(prompts_cfg["eval_system"], base_dir=config_dir))
    reflexion_prompt = ""
    if bool(pipeline_cfg.get("use_reflexion", False)):
        reflexion_prompt = load_text_file(
            resolve_path(prompts_cfg["reflexion_system"], base_dir=config_dir)
        )

    reason_model_cfg = models_cfg.get("reason") or {}
    eval_model_cfg = models_cfg.get("eval") or {}
    reflexion_model_cfg = models_cfg.get("reflexion") or reason_model_cfg

    # reason 支持 config 配 temperature（Mode1 重试需 temp>0 才有变化）
    reason_cfg_dict = ChatGPTConfig().as_dict()
    if reason_model_cfg.get("temperature") is not None:
        reason_cfg_dict["temperature"] = float(reason_model_cfg["temperature"])
    if reason_model_cfg.get("max_tokens") is not None:
        reason_cfg_dict["max_tokens"] = int(reason_model_cfg["max_tokens"])

    eval_cfg_dict = ChatGPTConfig().as_dict()
    if eval_model_cfg.get("temperature") is not None:
        eval_cfg_dict["temperature"] = float(eval_model_cfg["temperature"])
    if eval_model_cfg.get("max_tokens") is not None:
        eval_cfg_dict["max_tokens"] = int(eval_model_cfg["max_tokens"])

    reflexion_cfg_dict = ChatGPTConfig().as_dict()
    if reflexion_model_cfg.get("temperature") is not None:
        reflexion_cfg_dict["temperature"] = float(reflexion_model_cfg["temperature"])
    if reflexion_model_cfg.get("max_tokens") is not None:
        reflexion_cfg_dict["max_tokens"] = int(reflexion_model_cfg["max_tokens"])

    reason_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=reason_model_cfg["name"],
        url=get_base_url(reason_model_cfg),
        api_key=get_api_key(reason_model_cfg),
        model_config_dict=reason_cfg_dict,
        token_counter=StubTokenCounter(),
        timeout=tcfg["request_timeout"],
        max_retries=tcfg["max_retries"],
        **openai_compatible_client_kwargs(),
    )
    eval_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=eval_model_cfg["name"],
        url=get_base_url(eval_model_cfg),
        api_key=get_api_key(eval_model_cfg),
        model_config_dict=eval_cfg_dict,
        token_counter=StubTokenCounter(),
        timeout=tcfg["request_timeout"],
        max_retries=tcfg["max_retries"],
        **openai_compatible_client_kwargs(),
    )
    reflexion_model = None
    if bool(pipeline_cfg.get("use_reflexion", False)):
        reflexion_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=reflexion_model_cfg["name"],
            url=get_base_url(reflexion_model_cfg),
            api_key=get_api_key(reflexion_model_cfg),
            model_config_dict=reflexion_cfg_dict,
            token_counter=StubTokenCounter(),
            timeout=tcfg["request_timeout"],
            max_retries=tcfg["max_retries"],
            **openai_compatible_client_kwargs(),
        )

    agent_kwargs = dict(
        retry_attempts=tcfg["agent_retry_attempts"],
        retry_delay=tcfg["agent_retry_delay"],
        step_timeout=tcfg["agent_step_timeout"],
    )
    reason_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name=agents_cfg.get("reason_role_name", "Reconfiguration Expert"),
            content=reason_prompt,
        ),
        model=reason_model,
        message_window_size=10,
        **agent_kwargs,
    )
    evaluate_agent = ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name=agents_cfg.get("eval_role_name", "Reconfiguration Evaluator"),
            content=eval_prompt,
        ),
        model=eval_model,
        **agent_kwargs,
    )
    reflexion_agent = None
    if reflexion_model is not None:
        reflexion_agent = ChatAgent(
            system_message=BaseMessage.make_assistant_message(
                role_name=agents_cfg.get("reflexion_role_name", "Reconfiguration Reflection Agent"),
                content=reflexion_prompt,
            ),
            model=reflexion_model,
            **agent_kwargs,
        )
    return reason_agent, evaluate_agent, reflexion_agent


def main():
    args = parse_args()
    load_project_env(REPO_ROOT / ".env")
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = merge_config_with_args(load_yaml_config(config_path), args)

    data_cfg = cfg.get("data") or {}
    pipeline_cfg = cfg.get("pipeline") or {}
    prompts_cfg = cfg.get("prompts") or {}
    metadata_extra_keys: List[str] = list(cfg.get("preserve_metadata_keys") or [])

    cot_history_raw = data_cfg.get("cot_history_path")
    if cot_history_raw:
        cot_history_path = str(resolve_path(cot_history_raw, base_dir=config_dir))
    else:
        cot_history_path = ""
    qa_data_path = str(resolve_path(data_cfg["qa_data_path"], base_dir=config_dir))
    output_dir = str(resolve_path(data_cfg["output_dir"], base_dir=config_dir))

    few_shot_cfg = parse_few_shot_config(cfg, config_dir=config_dir)
    if args.few_shot_num_samples is not None:
        few_shot_cfg.num_samples = max(1, int(args.few_shot_num_samples))
    if args.few_shot_seed is not None:
        few_shot_cfg.seed = int(args.few_shot_seed)
    few_shot_pool = load_few_shot_examples(few_shot_cfg.path)
    effective_k = min(few_shot_cfg.num_samples, len(few_shot_pool))

    print(f"[INFO] 配置: {config_path}")
    print(
        f"[INFO] few_shot={few_shot_cfg.path} "
        f"(pool={len(few_shot_pool)}, per_problem_k={effective_k}, seed={few_shot_cfg.seed})"
    )
    if effective_k < few_shot_cfg.num_samples:
        print(
            f"[WARN] few_shot num_samples={few_shot_cfg.num_samples} 大于池大小 "
            f"{len(few_shot_pool)}，已降为 {effective_k}"
        )

    distilled = load_jsonl_records_optional(cot_history_path, source_name="cot_history_path")
    distilled_questions = set()
    for d in distilled:
        q = extract_question(d)
        if q:
            distilled_questions.add(q)
    print(
        f"历史蒸馏数据: {len(distilled)} 条总行, 唯一 question: {len(distilled_questions)}"
    )

    qa_data = load_jsonl_records(qa_data_path, source_name="qa_data_path")
    print(f"原始 qa_data 总量: {len(qa_data)}")

    deduped_records = []
    seen_new_questions = set()
    skipped_distilled = 0
    skipped_dup = 0

    for d in qa_data:
        question = extract_question(d)
        if not question:
            continue
        if question in distilled_questions:
            skipped_distilled += 1
            continue
        if question in seen_new_questions:
            skipped_dup += 1
            continue
        seen_new_questions.add(question)
        deduped_records.append(d)

    print(
        f"过滤后待蒸馏: {len(deduped_records)} | 已蒸馏: {skipped_distilled} | "
        f"批内重复: {skipped_dup}"
    )

    total_deduped = len(deduped_records)
    # --indices 优先：挑指定行（不连续），一个 batch 并发跑；否则用连续切片
    if args.indices:
        idx_list = [int(x) for x in args.indices.split(",") if x.strip() != ""]
        bad = [i for i in idx_list if not (0 <= i < total_deduped)]
        if bad:
            raise ValueError(f"indices 越界（去重后总量 {total_deduped}）: {bad}")
        selected_pairs = [(i, deduped_records[i]) for i in idx_list]
        print(f"指定索引: {idx_list}, 本次蒸馏 {len(selected_pairs)} 条（一个 batch 并发）")
    else:
        start_idx = max(0, int(args.qa_start))
        end_idx = total_deduped if int(args.qa_end) < 0 else min(total_deduped, int(args.qa_end))
        if start_idx >= end_idx:
            raise ValueError(
                f"非法区间: qa_start={start_idx}, qa_end={end_idx}, 去重后总量={total_deduped}"
            )
        selected_pairs = list(enumerate(deduped_records[start_idx:end_idx], start=start_idx))
        print(f"切片区间: [{start_idx}, {end_idx}), 本次蒸馏 {len(selected_pairs)} 条")

    problems = []
    for idx, d in selected_pairs:
        question = extract_question(d)
        solution = extract_solution(d)
        problems.append(
            {
                "problem": question,
                "solution": solution,
                "meta_data": build_sample_meta_data(
                    d,
                    source_index=idx,
                    extra_keys=metadata_extra_keys,
                ),
            }
        )

    print("最终送入蒸馏的样本数：", len(problems))
    if not problems:
        print("没有新增数据需要蒸馏，程序退出。")
        return

    timeout_cfg = parse_timeout_cfg(pipeline_cfg)
    print(
        "[INFO] 超时/重试配置: "
        f"request_timeout={timeout_cfg['request_timeout']}s, "
        f"max_retries={timeout_cfg['max_retries']}, "
        f"agent_step_timeout={timeout_cfg['agent_step_timeout']}s, "
        f"agent_retry_attempts={timeout_cfg['agent_retry_attempts']}, "
        f"problem_max_retries={timeout_cfg['problem_max_retries']}"
    )
    reason_agent, evaluate_agent, reflexion_agent = build_agents(
        cfg,
        config_dir,
        timeout_cfg=timeout_cfg,
    )
    print("agents 构造完毕")

    os.makedirs(output_dir, exist_ok=True)
    # 文件名：--indices 用 "indices" 标记 + 行数，否则用切片区间
    if args.indices:
        name_tag = f"indices{len(selected_pairs)}"
    else:
        name_tag = f"{start_idx}_{end_idx}"
    file_path = os.path.join(
        output_dir,
        f"generated_long_cot_{name_tag}_{formatted_datetime}.json",
    )

    user_templates = load_user_templates(prompts_cfg, config_dir=config_dir)

    pipeline = ReconfigCoTPipeline(
        reason_agent=reason_agent,
        evaluate_agent=evaluate_agent,
        problems=problems,
        max_iterations=int(pipeline_cfg.get("max_iterations", 1)),
        score_threshold=float(pipeline_cfg.get("score_threshold", 0.93)),
        output_path=file_path,
        batch_size=int(pipeline_cfg.get("batch_size", 64)),
        max_workers=int(pipeline_cfg.get("max_workers", 64)),
        few_shot_pool=few_shot_pool,
        few_shot_num_samples=few_shot_cfg.num_samples,
        few_shot_seed=few_shot_cfg.seed,
        use_reflexion=bool(pipeline_cfg.get("use_reflexion", False)),
        reflexion_agent=reflexion_agent,
        gt_correction_cfg=pipeline_cfg,
        quality_cfg=pipeline_cfg,
        timeout_cfg=timeout_cfg,
        user_templates=user_templates,
    )

    print("Start generation! May take some time, please wait..")
    run_pipeline_with_progress(
        pipeline=pipeline,
        problems=problems,
        rationalization=False,
        timeout_cfg=timeout_cfg,
    )

    print(f"数据构造完毕！{file_path}")


if __name__ == "__main__":
    main()
