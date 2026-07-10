"""Shared utilities for CoT distillation scripts."""

from __future__ import annotations

import errno
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

try:
    import yaml
except ImportError as e:
    raise ImportError("请安装 PyYAML: pip install pyyaml") from e

from utils.metrics_utils import (  # noqa: E402
    compute_gt_match,
    graph_penalties_from_open_lines,
    parse_open_lines,
    parse_open_lines_full_xml,
)

_BUS_EQ_RE = re.compile(r"busses?\s*=\s*(\d+)", re.IGNORECASE)


@dataclass
class FewShotConfig:
    path: Path
    num_samples: int = 2
    seed: int = 42


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


def get_api_key(cfg: dict) -> str:
    env_name = str(cfg.get("api_key_env") or "OPENAI_API_KEY")
    key = os.environ.get(env_name, "").strip()
    if not key:
        key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(f"未找到 API Key，请设置环境变量 {env_name} 或 OPENAI_API_KEY")
    return key


def get_base_url(cfg: dict) -> str:
    env_name = str(cfg.get("base_url_env") or "").strip()
    if env_name:
        url = os.environ.get(env_name, "").strip()
        if url:
            return url
    url = str(cfg.get("base_url") or "").strip()
    if not url:
        hint = f" 或环境变量 {env_name}" if env_name else ""
        raise ValueError(f"未找到 base_url，请设置 models.*.base_url{hint}")
    return url


def is_disk_full_error(exc: Exception) -> bool:
    cur = exc
    while cur is not None:
        if isinstance(cur, OSError) and cur.errno in (errno.ENOSPC, errno.EDQUOT):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def assemble_trace_from_message(msg) -> str:
    """Assemble reasoning_content and content into a <think> + answer trace."""
    open_tag = chr(60) + "think" + chr(62)
    close_tag = chr(60) + "/think" + chr(62)
    reasoning = (getattr(msg, "reasoning_content", None) or "").strip()
    content = (getattr(msg, "content", None) or "").strip()
    if reasoning:
        return open_tag + "\n" + reasoning + "\n" + close_tag + "\n" + content
    return content


def parse_few_shot_config(cfg: dict, *, config_dir: Path) -> FewShotConfig:
    """Parse the few_shot object from a CoT YAML config."""
    raw = cfg.get("few_shot")
    if not isinstance(raw, dict):
        raise ValueError("配置 few_shot 需为 {path, num_samples, seed} 对象")
    path_raw = raw.get("path")
    if not path_raw:
        raise ValueError("few_shot 对象需包含 path 字段")
    return FewShotConfig(
        path=resolve_path(str(path_raw), base_dir=config_dir),
        num_samples=max(1, int(raw.get("num_samples", 2))),
        seed=int(raw.get("seed", 42)),
    )


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


def format_few_shot_examples(examples: List[dict]) -> str:
    """Render few-shot examples into the current user-prompt text format."""
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


def bus_of(text: str) -> int | None:
    """Extract bus count from full problem text containing `Busses=<n>`."""
    t = text or ""
    m = _BUS_EQ_RE.search(t)
    return int(m.group(1)) if m else None


def message_content(msg) -> str:
    """Return stripped textual content from a CAMEL/OpenAI-compatible message."""
    return (getattr(msg, "content", None) or "").strip()


def parse_model_json_object(text: str) -> dict:
    """Parse a JSON object from model text, tolerating prose around the object."""
    if not text:
        return {}
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def clamp01(value: Any, default: float = 0.0) -> float:
    """Coerce a value to float and clamp it to [0, 1]."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def extract_xml_block(text: str, tag: str) -> str:
    """Extract the inner text of a simple XML-style block."""
    if not text:
        return ""
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def extract_think(trace: str) -> str:
    """Extract the reasoning text from <think>, or the prefix before <answer>."""
    think = extract_xml_block(trace, "think")
    if think:
        return think
    answer_match = re.search(r"<answer\b", trace or "", re.IGNORECASE)
    prefix = trace[: answer_match.start()] if answer_match else trace
    return (prefix or "").strip()


def extract_answer_block(trace: str) -> str:
    """Extract and rewrap the full <answer> block from a trace."""
    answer = extract_xml_block(trace, "answer")
    if not answer:
        return ""
    return f"<answer>\n{answer}\n</answer>"


def compute_verifier_metrics(
    problem: str,
    trace: str,
    solution: str = "",
    *,
    min_iou_accept: float = 0.75,
) -> dict:
    """Compute hard verifier metrics for a generated CoT trace."""
    gen_open = parse_open_lines_full_xml(trace)
    gt_open = parse_open_lines_full_xml(solution) or parse_open_lines(solution)
    penalties = dict(graph_penalties_from_open_lines(problem, gen_open))
    is_valid = (
        bool(gen_open)
        and penalties["invalid_edges"] == 0.0
        and penalties["cycles"] == 0.0
        and penalties["subgraphs"] == 0.0
    )
    exact, iou = compute_gt_match(gen_open, gt_open) if gt_open else (0.0, 0.0)
    gen_set = {tuple(sorted(e)) for e in gen_open}
    gt_set = {tuple(sorted(e)) for e in gt_open}
    hit = len(gen_set & gt_set)
    precision = hit / len(gen_set) if gen_set else 0.0
    recall = hit / len(gt_set) if gt_set else 0.0
    alignment = (
        1.0
        if iou >= min_iou_accept
        else max(0.0, iou / max(min_iou_accept, 1e-6))
    )
    copied_input = False
    input_open = parse_open_lines(problem)
    if input_open and gen_open:
        copied_input = compute_gt_match(gen_open, input_open)[0] == 1.0
    return {
        "format": 1.0 if bool(gen_open) else 0.0,
        "graph_validity": 1.0 if is_valid else 0.0,
        "answer_alignment": float(alignment),
        "iou_hidden": float(iou),
        "precision_hidden": float(precision),
        "recall_hidden": float(recall),
        "gt_exact_match": float(exact),
        "hit_count_hidden": int(hit),
        "pred_open_count": int(len(gen_set)),
        "target_open_count": int(len(gt_set)),
        "copied_input_open": 1.0 if copied_input else 0.0,
        "invalid_edges": float(penalties["invalid_edges"]),
        "cycles": float(penalties["cycles"]),
        "subgraphs": float(penalties["subgraphs"]),
    }


def load_jsonl_records_optional(path: str, *, source_name: str) -> list:
    """Load optional JSONL records; return [] when path is blank, missing, or empty."""
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


STANDARD_META_KEYS = (
    "sample_id",
    "bus",
    "source_file",
    "row_index",
    "split",
    "source_id",
)
TOP_LEVEL_COMPAT_META_KEYS = (
    "source_index",
    "bus",
    "sample_id",
    "source_id",
)


def _resolve_field(sample: dict, key: str) -> Any:
    if key in sample:
        return sample[key]
    meta = sample.get("meta")
    if isinstance(meta, dict) and key in meta:
        return meta[key]
    return None


def build_sample_meta_data(
    sample: dict,
    source_index: int,
    extra_keys: list | None = None,
) -> dict:
    """Build flat meta_data for a sample before generation."""
    meta_data: dict[str, Any] = {"source_index": source_index}
    nested = sample.get("meta")
    if isinstance(nested, dict):
        for key, value in nested.items():
            if value is not None:
                meta_data[key] = value

    keys = list(STANDARD_META_KEYS)
    for key in extra_keys or []:
        if key in ("source_index", "meta"):
            continue
        if key not in keys:
            keys.append(key)
    for key in keys:
        value = _resolve_field(sample, key)
        if value is not None:
            meta_data[key] = value
    return meta_data


def build_top_level_metadata(meta_data: dict) -> dict:
    """Keep common top-level metadata fields for downstream compatibility."""
    return {
        key: meta_data[key]
        for key in TOP_LEVEL_COMPAT_META_KEYS
        if key in meta_data and meta_data[key] is not None
    }
