"""Grid reconfiguration metrics: parsing, graph penalties, and eval IO."""

from __future__ import annotations

import ast
import csv
import re

from utils.generation_utils import extract_output_data


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_open_lines(text: str):
    match = re.search(r"Open Lines=\[(.*?)\]", text)
    if not match:
        return []
    cleaned = re.sub(r"[^0-9,() ]", "", match.group(1))
    cleaned = re.sub(r"\s+", "", cleaned).strip(",")
    try:
        return ast.literal_eval(f"[{cleaned}]")
    except (ValueError, SyntaxError):
        return []


def _validate_edge_list(parsed):
    if not isinstance(parsed, list) or not parsed:
        return []

    open_lines = []
    for item in parsed:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            return []
        try:
            u, v = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            return []
        open_lines.append((u, v))
    return open_lines


def parse_open_lines_xml(text: str):
    """Strictly parse XML-only Open Lines output.

    Expected shape:
      <answer>
      <open_lines>
      [(u1,v1),(u2,v2)]
      </open_lines>
      </answer>

    Any extra answer/open_lines tag, missing tag, malformed edge list, or
    non-whitespace text after </answer> is treated as parse failure.
    """
    if not text:
        return []

    if len(re.findall(r"<answer>", text)) != 1 or len(re.findall(r"</answer>", text)) != 1:
        return []
    if len(re.findall(r"<open_lines>", text)) != 1 or len(re.findall(r"</open_lines>", text)) != 1:
        return []

    pattern = re.compile(
        r"^\s*<answer>\s*<open_lines>\s*(.*?)\s*</open_lines>\s*</answer>\s*$",
        re.DOTALL,
    )
    match = pattern.match(text)
    if not match:
        return []

    try:
        parsed = ast.literal_eval(match.group(1).strip())
    except (ValueError, SyntaxError):
        return []
    return _validate_edge_list(parsed)


def parse_open_lines_full_xml(text: str):
    """Strictly parse full XML output and return its Open Lines.

    Expected shape:
      <answer>
      <open_lines>...</open_lines>
      <node_voltages>...</node_voltages>
      <system_loss>...</system_loss>
      </answer>

    Only Open Lines are returned because reward/eval compare topology. The
    remaining tags are required as a format gate for full-output XML ablations.
    """
    if not text:
        return []

    required_tags = ("answer", "open_lines", "node_voltages", "system_loss")
    for tag in required_tags:
        if len(re.findall(fr"<{tag}>", text)) != 1 or len(re.findall(fr"</{tag}>", text)) != 1:
            return []

    pattern = re.compile(
        r"^\s*<answer>\s*"
        r"<open_lines>\s*(.*?)\s*</open_lines>\s*"
        r"<node_voltages>\s*(.*?)\s*</node_voltages>\s*"
        r"<system_loss>\s*(.*?)\s*</system_loss>\s*"
        r"</answer>\s*$",
        re.DOTALL,
    )
    match = pattern.match(text)
    if not match:
        return []

    try:
        parsed_open = ast.literal_eval(match.group(1).strip())
        parsed_voltages = ast.literal_eval(match.group(2).strip())
        float(match.group(3).strip())
    except (ValueError, SyntaxError):
        return []

    if not isinstance(parsed_voltages, list) or not parsed_voltages:
        return []
    try:
        [float(value) for value in parsed_voltages]
    except (TypeError, ValueError):
        return []

    return _validate_edge_list(parsed_open)


def parse_available_lines(text: str):
    match = re.search(r"Lines=\[(.*?)\]", text)
    if not match:
        return []
    try:
        return ast.literal_eval(f"[{match.group(1)}]")
    except (ValueError, SyntaxError):
        return []


def parse_num_buses(text: str) -> int:
    match = re.search(r"Busses=(\d+)", text or "")
    return int(match.group(1)) if match else 0


def get_output_graph_edges(predicted_lines, available_lines):
    open_lines = [tuple(line) for line in predicted_lines]
    open_lines += [tuple(line)[::-1] for line in open_lines]
    return [tuple(line) for line in available_lines if tuple(line) not in open_lines]


# ---------------------------------------------------------------------------
# Graph penalty computation
# ---------------------------------------------------------------------------

def compute_invalid_edges_loss(predicted_lines, available_lines):
    available_set = {tuple(line) for line in available_lines}
    available_set |= {(b, a) for a, b in available_set}
    invalid_edges_loss = 0.0
    for line in predicted_lines:
        if tuple(line) not in available_set:
            invalid_edges_loss += 1.0
    return invalid_edges_loss


def compute_cycles_loss(graph_edges):
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(graph_edges)
    cycles_loss = 0.0
    try:
        cycles_loss += len(list(nx.find_cycle(graph, orientation="ignore")))
    except nx.NetworkXNoCycle:
        pass
    return cycles_loss


def compute_subgraphs_loss(graph_edges):
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(graph_edges)
    if graph.number_of_nodes() == 0:
        return 1.0
    subgraphs_loss = 0.0
    connected_components = nx.number_connected_components(graph)
    if connected_components > 1:
        subgraphs_loss += connected_components - 1
    return subgraphs_loss


def _canonical_edge_set(lines):
    """Normalize a list of edges to an undirected set of (min, max) int tuples.

    Drops malformed entries (non-pair, non-int) silently — caller already
    decides what "no edges" means.
    """
    out = set()
    for line in lines or []:
        if not isinstance(line, (tuple, list)) or len(line) != 2:
            continue
        try:
            a, b = int(line[0]), int(line[1])
        except (TypeError, ValueError):
            continue
        out.add((min(a, b), max(a, b)))
    return out


def compute_gt_match(predicted_lines, correct_lines) -> tuple[float, float]:
    """Undirected exact-match + IoU between predicted and ground-truth Open Lines.

    Returns (exact_match, iou). exact_match=1.0 iff sets are equal. IoU=1.0
    if both are empty (vacuously equal); 0.0 only if union is empty after one
    side is non-empty (cannot happen by set algebra, kept for clarity).
    """
    p = _canonical_edge_set(predicted_lines)
    g = _canonical_edge_set(correct_lines)
    exact = 1.0 if p == g else 0.0
    union = p | g
    iou = (len(p & g) / len(union)) if union else 1.0
    return exact, iou


def graph_penalties(
    input_text: str, output_text: str, *, normalize: bool = False
) -> dict[str, float]:
    """Compute normalized penalty components (used by RL reward and SFT custom loss)."""
    predicted_lines = parse_open_lines(output_text)
    return graph_penalties_from_open_lines(input_text, predicted_lines, normalize=normalize)


def graph_penalties_from_open_lines(
    input_text: str, predicted_lines, *, normalize: bool = False
) -> dict[str, float]:
    """Compute graph validity penalties from already parsed Open Lines.

    When normalize=True, penalties are divided by network size to be comparable
    across networks of different scales (see Eq. 5 in the paper).
    """
    available_lines = parse_available_lines(input_text)
    num_buses = parse_num_buses(input_text)
    if predicted_lines and not all(
        isinstance(line, (tuple, list)) and len(line) == 2 for line in predicted_lines
    ):
        predicted_lines = []

    all_edges = _canonical_edge_set(available_lines)
    open_edges = _canonical_edge_set(predicted_lines)
    if num_buses <= 0 and all_edges:
        num_buses = max(node for edge in all_edges for node in edge)

    if not predicted_lines or not all_edges or num_buses <= 0:
        return {"invalid_edges": 1.0, "cycles": 1.0, "subgraphs": 1.0}

    import networkx as nx

    invalid_edges = len(open_edges - all_edges)
    closed_edges = all_edges - open_edges

    graph = nx.Graph()
    graph.add_nodes_from(range(1, num_buses + 1))
    graph.add_edges_from(closed_edges)

    connected_components = nx.number_connected_components(graph)
    subgraphs = max(0, connected_components - 1)
    cycles = max(0, len(closed_edges) - num_buses + connected_components)

    if normalize:
        return {
            "invalid_edges": float(invalid_edges) / max(1, len(all_edges)),
            "cycles": float(cycles) / max(1, num_buses),
            "subgraphs": float(subgraphs) / max(1, num_buses),
        }
    return {
        "invalid_edges": float(invalid_edges),
        "cycles": float(cycles),
        "subgraphs": float(subgraphs),
    }


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def get_number_of_nodes(available_lines):
    nodes = [node for line in available_lines for node in line]
    return max(nodes) if nodes else 0


def extract_metrics(available_lines, reformatted_response):
    num_nodes = get_number_of_nodes(available_lines)
    generated_open_lines = reformatted_response["Open Lines"]
    generated_node_voltages = reformatted_response["Node Voltages"]
    system_loss = reformatted_response["System Loss"]
    return num_nodes, generated_open_lines, generated_node_voltages, system_loss


def parse_correct_output(correct_output):
    correct_data = extract_output_data(correct_output)
    correct_open_lines = correct_data["Open Lines"]
    correct_generated_lines = correct_data["Node Voltages"]
    correct_system_loss = correct_data["System Loss"]
    return correct_open_lines, correct_generated_lines, correct_system_loss


# ---------------------------------------------------------------------------
# Eval CSV / TXT IO
# ---------------------------------------------------------------------------

def prep_csv(filename: str, columns: list[str]) -> None:
    """Create CSV with header row = `columns`."""
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)


def write_to_csv(filename: str, rows: list[dict], columns: list[str]) -> None:
    """Append `rows` (list of dicts) to `filename`, ordered by `columns`.

    Missing keys serialize as empty string, so partial rows are safe.
    """
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        for row in rows:
            writer.writerow([row.get(c, "") for c in columns])


def write_to_txt(
    filename: str,
    sections: list[tuple[str, list[tuple[str, str]]]],
) -> None:
    """Write a metrics text file with grouped [Section] / label: value layout.

    `sections` is a list of (section_name, items) where items is a list of
    (label, value_string) pairs. The same structure is used by stdout
    printing so the on-disk file matches what the user sees.
    """
    with open(filename, "w") as file:
        file.write("---- Evaluation Metrics ----\n")
        for section_name, items in sections:
            file.write(f"\n[{section_name}]\n")
            for label, value in items:
                file.write(f"  {label}: {value}\n")
