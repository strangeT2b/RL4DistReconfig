# CoT 数据蒸馏（配置外置：config/*.yaml + prompts/ + few_shot/）
from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.config_utils import load_project_env  # noqa: E402

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.datagen import SelfImprovingCoTPipeline
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.models.stub_model import StubTokenCounter
from camel.types import ModelPlatformType
from tqdm import tqdm

try:
    import yaml
except ImportError as e:
    raise ImportError("请安装 PyYAML: pip install pyyaml") from e

DISTILL_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = DISTILL_DIR / "config" / "reconfig_short.yaml"

formatted_datetime = datetime.now().strftime("%m-%d-%H:%M:%S")

# 并行 batch 内 per-problem few-shot（线程局部，避免多线程覆盖）
_few_shot_ctx = threading.local()


@dataclass
class FewShotConfig:
    path: Path
    num_samples: int = 2
    seed: int = 42


def parse_few_shot_config(cfg: dict, *, config_dir: Path) -> FewShotConfig:
    """解析 few_shot：兼容旧版字符串路径与新版对象配置。"""
    raw = cfg.get("few_shot")
    if raw is None:
        raise ValueError("配置缺少 few_shot（路径字符串或 {path, num_samples, seed} 对象）")
    if isinstance(raw, str):
        path = resolve_path(raw, base_dir=config_dir)
        return FewShotConfig(path=path, num_samples=2, seed=42)
    if isinstance(raw, dict):
        path_raw = raw.get("path") or raw.get("file")
        if not path_raw:
            raise ValueError("few_shot 对象需包含 path 字段")
        return FewShotConfig(
            path=resolve_path(str(path_raw), base_dir=config_dir),
            num_samples=max(1, int(raw.get("num_samples", 2))),
            seed=int(raw.get("seed", 42)),
        )
    raise ValueError(f"few_shot 配置类型不支持: {type(raw)!r}")


def format_few_shot_examples(examples: List[dict]) -> str:
    """将 few-shot 列表格式化为注入 REASONING_TEMPLATE 的可读文本。"""
    blocks: List[str] = []
    for i, ex in enumerate(examples, 1):
        problem = str(ex.get("problem", "")).strip()
        solution = str(ex.get("solution", "")).strip()
        style_tag = ex.get("reasoning_style") or ex.get("style") or ""
        header = f"### Exemplar {i}"
        if style_tag:
            header += f" (style: {style_tag})"
        blocks.append(
            f"{header}\n"
            f"Problem:\n{problem}\n\n"
            "Reference solution (learn the reasoning style and depth only; "
            "do not copy its exact open_lines, numbers, or voltages verbatim — "
            "your answer must fit the current problem):\n"
            f"{solution}"
        )
    return "\n\n".join(blocks)


def sample_few_shot_examples(
    pool: List[dict],
    k: int,
    *,
    seed: str,
) -> List[dict]:
    """按 problem 文本派生种子，保证同题可复现、不同题随机。"""
    if not pool or k <= 0:
        return []
    k = min(k, len(pool))
    rng = random.Random(f"{seed}:{k}")
    if k >= len(pool):
        return list(pool)
    return rng.sample(pool, k)


_BUS_EQ_RE = re.compile(r"busses?\s*=\s*(\d+)", re.IGNORECASE)


def bus_of(text: str) -> int | None:
    """从 full problem 文本提取 bus 数；few-shot 必须包含 `Busses=<n>`。"""
    t = text or ""
    m = _BUS_EQ_RE.search(t)
    return int(m.group(1)) if m else None


class ReconfigCoTPipeline(SelfImprovingCoTPipeline):
    """配电网重构 CoT 蒸馏 pipeline：per-problem 采样 few-shot（线程安全）。"""

    # 薄壳：只塞 problem + few-shot；规则在 system message (prompts/reason_system.txt) 里，不在此重复。
    REASONING_TEMPLATE = (
        "Solve the distribution network reconfiguration problem below using "
        "the structured method in your instructions.\n\n"
        "{few_shot_examples}"
        "Problem:\n{problem}\n\n"
        "Produce the reasoning block and the <answer> block."
    )

    # 覆盖 camel 默认 IMPROVEMENT_TEMPLATE：默认"improve this trace"→模型验证式。
    # 改成"禁止验证、从头推导为什么该解降损"，逼模型 DERIVE 而非 verify。
    # 仍含 {solution}=GT（STaR：据 GT 补推导），{trace}/{feedback} 作 scaffolding 强制产出推理。
    IMPROVEMENT_TEMPLATE = (
        "You are given a distribution network reconfiguration problem and its VERIFIED optimal solution. "
        "Produce a complete FIRST-PRINCIPLES DERIVATION of why this solution minimizes system loss.\n\n"
        "Do NOT verify or check whether the solution forms a valid tree — that is not the task. "
        "Do NOT merely restate or justify the given answer. Instead DERIVE it from the load data:\n"
        "- Identify the dominant heavy loads (largest |S|) and their bus locations.\n"
        "- For each, trace its current feed path from the source and explain why that path is high-loss "
        "(long / high-impedance, carrying heavy downstream current).\n"
        "- Step by step, show why opening exactly the lines in the solution gives those heavy loads "
        "shorter feed paths, reducing total I^2R loss.\n"
        "- The solution's open_lines must be the CONCLUSION of your derivation, not the premise.\n\n"
        "Problem:\n{problem}\n\n{solution}\n\n"
        "Previous reasoning trace (reference only; may be wrong — re-derive, do not patch):\n{trace}\n\n"
        "Feedback: {feedback}\n\n"
        "Produce the full derivation, then output the <answer> block."
    )

    def __init__(
        self,
        *,
        few_shot_pool: List[dict],
        few_shot_num_samples: int,
        few_shot_seed: int,
        timeout_cfg: Optional[dict] = None,
        **kwargs: Any,
    ):
        # 父类 few_shot_examples 仅作占位；实际由本类 per-problem 注入
        super().__init__(few_shot_examples=None, **kwargs)
        self.few_shot_pool = few_shot_pool
        self.few_shot_num_samples = max(1, few_shot_num_samples)
        self.few_shot_seed = few_shot_seed
        self._timeout_cfg = parse_timeout_cfg(timeout_cfg or {})

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
        seed_key = f"{self.few_shot_seed}:{problem_text}"
        picked = sample_few_shot_examples(
            pool,
            self.few_shot_num_samples,
            seed=seed_key,
        )
        return format_few_shot_examples(picked)

    def process_problem(
        self, problem: Dict, rationalization: bool = False
    ):
        _few_shot_ctx.examples_str = self._sample_few_shot_str(problem)
        return super().process_problem(problem, rationalization=rationalization)

    def _active_few_shot_block(self) -> str:
        examples_str = getattr(_few_shot_ctx, "examples_str", "") or ""
        if not examples_str:
            return ""
        return f"Examples:\n{examples_str}"

    @staticmethod
    def _assemble_trace(msg) -> str:
        # gateway 把推理分到 reasoning_content、content 只剩 <answer>；
        # 这里把两者拼成 think-open + reasoning + think-close + content，
        # 让 trace 带推理块。开闭标签用 chr() 构造，避免字面量被序列化层
        # 破坏（见 think-token-literal-mangling）。
        _open = chr(60) + "think" + chr(62)
        _close = chr(60) + "/think" + chr(62)
        reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
        content = (getattr(msg, "content", None) or "").strip()
        if reasoning:
            return _open + "\n" + reasoning + "\n" + _close + "\n" + content
        return content

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
        best_avg_score = 0.01
        candidate_avg_scores: List[float] = []
        for trace in candidate_traces:
            eval_results = self.evaluate_trace(problem, trace)
            scores = {k: v for k, v in eval_results.items() if k != "feedback"}
            avg_score = sum(scores.values()) / len(scores) if scores else 0.0
            candidate_avg_scores.append(avg_score)
            if self._check_score_threshold(scores) and avg_score > best_avg_score:
                best_trace = trace
                best_avg_score = avg_score
        if best_trace is None and candidate_traces:
            best_trace = candidate_traces[
                candidate_avg_scores.index(max(candidate_avg_scores))
            ]
        return best_trace or ""

    def improve_trace(self, problem, trace, feedback, solution=None):
        # 覆盖 camel 默认：它 return response.msg.content 会丢 reasoning_content；
        # 用 _assemble_trace 拼回 think 推理块。IMPROVEMENT_TEMPLATE 已覆盖成推导式（禁验证）。
        # 加重试：reasoning 为空（模型 echo GT 答案）就重采（需 temp>0 才有变化），逼出推理。
        self.reason_agent.reset()
        solution_text = f"Solution: {solution}" if solution else ""
        prompt = self.IMPROVEMENT_TEMPLATE.format(
            problem=problem,
            trace=trace,
            feedback=feedback,
            solution=solution_text,
        )
        OPEN = chr(60) + "think" + chr(62)
        max_retry = self._timeout_cfg.get("improve_retry", 5)
        assembled = ""
        for attempt in range(max_retry):
            response = self.reason_agent.step(prompt)
            assembled = self._assemble_trace(response.msg)
            if OPEN in assembled:  # 产出了 think 推理块
                return assembled
            self.reason_agent.reset()  # 空 reasoning，重采
        return assembled  # 重试用尽仍空，返回最后一条（至少有 answer）


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


def _is_disk_full_error(exc: Exception) -> bool:
    cur = exc
    while cur is not None:
        if isinstance(cur, OSError) and cur.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def run_pipeline_with_progress(
    pipeline,
    problems,
    rationalization=True,
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
                if _is_disk_full_error(e):
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
                        if _is_disk_full_error(e):
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


def resolve_path(raw: str, *, base_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件必须是 YAML 对象: {path}")
    return cfg


def load_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_few_shot_examples(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"few_shot 文件必须是 JSON 数组: {path}")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "problem" not in item or "solution" not in item:
            raise ValueError(f"few_shot[{i}] 需包含 problem 与 solution 字段")
        if bus_of(str(item.get("problem", ""))) is None:
            raise ValueError(f"few_shot[{i}] problem 缺少 Busses=<n> 字段，无法做 cross-bus 采样")
    return data


def get_api_key(cfg: dict) -> str:
    env_name = str(cfg.get("api_key_env") or "OPENAI_API_KEY")
    key = os.environ.get(env_name, "").strip()
    if not key:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(f"未找到 API Key，请设置环境变量 {env_name} 或 OPENAI_API_KEY")
    return key


def merge_config_with_args(cfg: dict, args: argparse.Namespace) -> dict:
    """CLI 显式传入的非 None 值覆盖 YAML。"""
    out = json.loads(json.dumps(cfg))  # deep copy

    if args.task_type is not None:
        out["task_type"] = args.task_type
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
        if isinstance(fs, dict):
            if args.few_shot_num_samples is not None:
                fs["num_samples"] = args.few_shot_num_samples
            if args.few_shot_seed is not None:
                fs["seed"] = args.few_shot_seed
        else:
            merged = {"path": fs or "../few_shot/long_few_shot.json"}
            if args.few_shot_num_samples is not None:
                merged["num_samples"] = args.few_shot_num_samples
            if args.few_shot_seed is not None:
                merged["seed"] = args.few_shot_seed
            out["few_shot"] = merged
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="CoT distillation runner（YAML 配置 + 任务可切换）")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML 配置文件路径（默认 CoT_distill/config/reconfig_short.yaml）",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="覆盖配置中的 task_type，仅蒸馏该任务样本（与 GT 行 task_type 匹配，忽略大小写）",
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
    return parser.parse_args()


def load_jsonl_records_optional(path: str, *, source_name: str) -> list:
    """历史 CoT 文件可选：不存在或为空时返回 []，不抛错，等价于不做历史去重。"""
    if not path or not str(path).strip():
        print(f"[INFO] 未配置 {source_name}，不进行历史去重")
        return []
    p = Path(path)
    if not p.is_file():
        print(f"[INFO] {source_name} 文件不存在，不进行历史去重: {p}")
        return []
    if p.stat().st_size == 0:
        print(f"[INFO] {source_name} 为空文件，不进行历史去重: {p}")
        return []
    return load_jsonl_records(str(p), source_name=source_name)


def load_jsonl_records(path: str, *, source_name: str) -> list:
    records = []
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                bad_lines += 1
                print(f"[WARN] 跳过 {source_name} 非法 JSON 行: line={line_no}, err={e}")
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                bad_lines += 1
                print(f"[WARN] 跳过 {source_name} 非对象 JSON 行: line={line_no}")
    if bad_lines:
        print(f"[INFO] {source_name} 跳过异常行数: {bad_lines}")
    return records


def extract_question(sample: dict) -> str:
    if not isinstance(sample, dict):
        return ""
    if sample.get("question"):
        return str(sample["question"]).strip()
    messages = sample.get("messages")
    if isinstance(messages, list):
        if len(messages) > 1 and isinstance(messages[1], dict):
            content = messages[1].get("content", "")
            return str(content).strip() if content is not None else ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content", "")
                return str(content).strip() if content is not None else ""
    return ""


def extract_solution(sample: dict) -> str:
    if not isinstance(sample, dict):
        return ""
    if sample.get("answer"):
        return str(sample["answer"]).strip()
    messages = sample.get("messages")
    if isinstance(messages, list):
        if len(messages) > 2 and isinstance(messages[2], dict):
            content = messages[2].get("content", "")
            return str(content).strip() if content is not None else ""
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content", "")
                return str(content).strip() if content is not None else ""
    return ""


def normalize_task_type_name(task_type: str) -> str:
    return str(task_type or "").strip().lower().replace("_", "-")


def get_sample_task_type(sample: dict) -> str:
    meta = sample.get("meta") if isinstance(sample.get("meta"), dict) else {}
    return str(sample.get("task_type") or meta.get("task_family") or "").strip()


def sample_matches_task_type(sample: dict, target_task_type: str) -> bool:
    if not target_task_type:
        return True
    raw = get_sample_task_type(sample)
    if not raw:
        return False
    return normalize_task_type_name(raw) == normalize_task_type_name(target_task_type)


def count_task_types(records: list) -> Counter:
    c: Counter = Counter()
    for d in records:
        if not isinstance(d, dict):
            continue
        raw = get_sample_task_type(d)
        c[raw or "(missing)"] += 1
    return c


# 写入 trace 顶层与 meta_data 的元数据键；供下游使用
COMMON_METADATA_KEYS = (
    "source_index",
    "task_type",
    "meta",
)


def _resolve_field(sample: dict, key: str) -> Any:
    if key in sample:
        return sample[key]
    meta = sample.get("meta")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return None


def build_preserved_metadata(sample: dict, source_index: int, preserve_keys: list) -> dict:
    keys: List[str] = list(preserve_keys or [])
    for k in COMMON_METADATA_KEYS:
        if k not in keys and k != "source_index":
            keys.append(k)

    preserved: Dict[str, Any] = {"source_index": source_index}
    for key in keys:
        if key == "source_index":
            continue
        val = _resolve_field(sample, key)
        if val is not None:
            preserved[key] = val

    if preserved.get("task_type") is None:
        preserved["task_type"] = get_sample_task_type(sample)

    return preserved


def build_meta_data_blob(preserved: dict) -> dict:
    """把 preserved（含嵌套 meta）扁平化成 meta_data，供下游消费。"""
    meta_data: Dict[str, Any] = {}
    for key in COMMON_METADATA_KEYS:
        if key in preserved and preserved[key] is not None:
            meta_data[key] = preserved[key]
    nested = preserved.get("meta")
    if isinstance(nested, dict):
        for k, v in nested.items():
            if k not in meta_data and v is not None:
                meta_data[k] = v
    return meta_data


def _load_generated_records(output_path: str):
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        lines = [json.loads(ln) for ln in content.splitlines() if ln.strip()]
        return lines


def enrich_generated_file_with_metadata(output_path: str, problem_to_metadata: dict) -> tuple:
    data = _load_generated_records(output_path)
    if data is None:
        return 0, 0

    if isinstance(data, dict):
        records = []
        for key in ("traces", "data", "records"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
    elif isinstance(data, list):
        records = data
    else:
        records = []

    enriched = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        problem = rec.get("problem") or rec.get("question")
        if not isinstance(problem, str):
            continue
        meta = problem_to_metadata.get(problem)
        if meta is None:
            continue
        for k, v in meta.items():
            rec[k] = v
        meta_data = build_meta_data_blob(meta)
        if meta_data:
            rec["meta_data"] = meta_data
        enriched += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return enriched, len(records)


def build_agents(cfg: dict, config_dir: Path, *, timeout_cfg: Optional[dict] = None):
    agents_cfg = cfg.get("agents") or {}
    prompts_cfg = cfg.get("prompts") or {}
    models_cfg = cfg.get("models") or {}
    tcfg = parse_timeout_cfg(timeout_cfg or {})

    reason_prompt = load_text_file(resolve_path(prompts_cfg["reason_system"], base_dir=config_dir))
    eval_prompt = load_text_file(resolve_path(prompts_cfg["eval_system"], base_dir=config_dir))

    reason_model_cfg = models_cfg.get("reason") or {}
    eval_model_cfg = models_cfg.get("eval") or {}

    # reason 支持 config 配 temperature（Mode1 重试需 temp>0 才有变化）
    reason_cfg_dict = ChatGPTConfig().as_dict()
    if reason_model_cfg.get("temperature") is not None:
        reason_cfg_dict["temperature"] = float(reason_model_cfg["temperature"])

    reason_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=reason_model_cfg["name"],
        url=reason_model_cfg["base_url"],
        api_key=get_api_key(reason_model_cfg),
        model_config_dict=reason_cfg_dict,
        token_counter=StubTokenCounter(),
        timeout=tcfg["request_timeout"],
        max_retries=tcfg["max_retries"],
    )
    eval_model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=eval_model_cfg["name"],
        url=eval_model_cfg["base_url"],
        api_key=get_api_key(eval_model_cfg),
        token_counter=StubTokenCounter(),
        timeout=tcfg["request_timeout"],
        max_retries=tcfg["max_retries"],
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
    return reason_agent, evaluate_agent


def main():
    args = parse_args()
    load_project_env(REPO_ROOT / ".env")
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg = merge_config_with_args(load_yaml_config(config_path), args)

    task_type = str(cfg.get("task_type") or "reconfig").strip()
    data_cfg = cfg.get("data") or {}
    filter_cfg = cfg.get("filter") or {}
    pipeline_cfg = cfg.get("pipeline") or {}
    preserve_keys: List[str] = list(cfg.get("preserve_metadata_keys") or [])
    if "meta" not in preserve_keys:
        preserve_keys.append("meta")

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
        f"[INFO] task_type={task_type!r}, few_shot={few_shot_cfg.path} "
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
        if not sample_matches_task_type(d, task_type):
            continue
        q = extract_question(d)
        if q:
            distilled_questions.add(q)
    print(
        f"历史蒸馏数据(task={task_type}): {len(distilled)} 条总行, "
        f"匹配 task 且唯一 question: {len(distilled_questions)}"
    )

    qa_data = load_jsonl_records(qa_data_path, source_name="qa_data_path")
    print(f"原始 qa_data 总量: {len(qa_data)}（混合多任务 jsonl，将先按 task_type 过滤）")
    qa_task_dist = count_task_types(qa_data)
    print(f"qa_data task_type 分布: {dict(sorted(qa_task_dist.items(), key=lambda x: -x[1]))}")
    print(f"目标蒸馏 task_type={task_type!r}（配置 require_task_type_match 见 filter）")

    require_task = bool(filter_cfg.get("require_task_type_match", True))

    deduped_records = []
    seen_new_questions = set()
    skipped_task = 0
    skipped_distilled = 0
    skipped_dup = 0

    for d in qa_data:
        question = extract_question(d)
        if not question:
            continue
        if require_task and not sample_matches_task_type(d, task_type):
            skipped_task += 1
            continue
        if question in distilled_questions:
            skipped_distilled += 1
            continue
        if question in seen_new_questions:
            skipped_dup += 1
            continue
        seen_new_questions.add(question)
        deduped_records.append(d)

    matched_task = sum(1 for d in qa_data if sample_matches_task_type(d, task_type))
    print(
        f"task_type 匹配 {task_type!r}: {matched_task}/{len(qa_data)} 条"
    )
    print(
        f"过滤后待蒸馏: {len(deduped_records)} | 跳过 task_type≠{task_type}: {skipped_task} | "
        f"已蒸馏: {skipped_distilled} | 批内重复: {skipped_dup}"
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
    problem_to_metadata = {}
    for idx, d in selected_pairs:
        question = extract_question(d)
        solution = extract_solution(d)
        problems.append({"problem": question, "solution": solution})
        problem_to_metadata[question] = build_preserved_metadata(
            d, source_index=idx, preserve_keys=preserve_keys
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
    reason_agent, evaluate_agent = build_agents(cfg, config_dir, timeout_cfg=timeout_cfg)
    print("agents 构造完毕")

    os.makedirs(output_dir, exist_ok=True)
    # 文件名：--indices 用 "indices" 标记 + 行数，否则用切片区间
    if args.indices:
        name_tag = f"indices{len(selected_pairs)}"
    else:
        name_tag = f"{start_idx}_{end_idx}"
    file_path = os.path.join(
        output_dir,
        f"generated_cot_data_{task_type}_{name_tag}_{formatted_datetime}.json",
    )

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
        timeout_cfg=timeout_cfg,
    )

    print("Start generation! May take some time, please wait..")
    run_pipeline_with_progress(
        pipeline=pipeline,
        problems=problems,
        rationalization=True,
        timeout_cfg=timeout_cfg,
    )

    enriched_cnt, total_generated = enrich_generated_file_with_metadata(
        output_path=file_path,
        problem_to_metadata=problem_to_metadata,
    )
    print(
        f"元数据回填完成: enriched={enriched_cnt}/{total_generated}, "
        f"preserve_keys={preserve_keys}（含 meta）"
    )
    print(f"数据构造完毕！{file_path}")


if __name__ == "__main__":
    main()
