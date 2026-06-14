#!/usr/bin/env python3
"""Build XML processed training CSVs from unprocessed grid samples."""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import numpy as np


DEFAULT_BUS_SYSTEMS = (33, 69, 84)
DEFAULT_UNPROCESSED_DIR = Path("Dataset/Unprocessed")
DEFAULT_OUTPUT_CSV = Path("Dataset/Processed_xml/train_33_69_84_nodes_open_lines_xml.csv")

BASE_TASK_DESCRIPTION = """Find the optimal configuration, i.e. the optimal connectivity and optimal open lines of these buses and lines
so as to ensure energy distribution to the whole system while minimizing the power loss. The number given for the busses indicates the
total number of busses starting from 1 going all the way to the given number in increments of 1. Make sure the Open Lines
in the output include ONLY Lines that are given in the input and that you take into account their given properties.
The Available Lines WITHOUT the Open Lines should form a network graph that is a single graph, i.e. no subgraphs or
multiple connected components lists and the graph should NOT contain any cycles i.e. the number of available lines WITHOUT
the number of open lines should EQUAL the number of busses minus one. If you predict the system loss and the value is greater
than the current system loss, DO NOT reconfigure the network and return the same configuration as in the input. ONLY
return a reconfiguration if and only if the system loss you predict is less than the original one since that is the ultimate goal."""

OPEN_LINES_XML_OUTPUT_INSTRUCTION = """Return only the following XML format. Do not include Node Voltages, System Loss, explanations, markdown, or any text after </answer>.

<answer>
<open_lines>
[(u1,v1),(u2,v2),...]
</open_lines>
</answer>"""

FULL_OUTPUT_XML_INSTRUCTION = """Return only the following XML format. Do not include explanations, markdown, or any text after </answer>.

<answer>
<open_lines>
[(u1,v1),(u2,v2),...]
</open_lines>
<node_voltages>
[v1,v2,...]
</node_voltages>
<system_loss>
loss
</system_loss>
</answer>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create XML processed train CSVs from Dataset/Unprocessed samples."
    )
    parser.add_argument("--unprocessed_dir", type=Path, default=DEFAULT_UNPROCESSED_DIR)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--output_mode",
        choices=["open_lines", "full_output"],
        default="open_lines",
        help="open_lines keeps the current XML-only target; full_output also includes node voltages and system loss.",
    )
    parser.add_argument(
        "--bus_systems",
        nargs="+",
        type=int,
        default=list(DEFAULT_BUS_SYSTEMS),
        help="Bus systems to include, e.g. --bus_systems 33 69 84.",
    )
    return parser.parse_args()


def parse_literal(value: str):
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"malformed literal: {value!r}") from exc


def format_input(row: dict[str, str]) -> str:
    return (
        f"Power Distribution Network: Busses={row['buses']}, Lines={row['lines']}, "
        f"Line Impedances={row['line_impedances']}, Open Lines={row['existing_open_lines']}\n"
        f"Network Variables: NodeVoltages={row['existing_node_voltages']}, "
        f"System Loss={row['existing_system_loss']}, System Load={row['system_load']}\n"
    )


def task_description(output_mode: str) -> str:
    instruction = (
        FULL_OUTPUT_XML_INSTRUCTION
        if output_mode == "full_output"
        else OPEN_LINES_XML_OUTPUT_INSTRUCTION
    )
    return f"{BASE_TASK_DESCRIPTION}\n\n{instruction}"


def format_open_lines(open_lines: list[tuple[int, int]]) -> str:
    return "[" + ",".join(f"({u},{v})" for u, v in open_lines) + "]"


def format_float_list(values: list[float]) -> str:
    return "[" + ",".join(f"{value:g}" for value in values) + "]"


def format_open_lines_xml_output(open_lines: list[tuple[int, int]]) -> str:
    return (
        "<answer>\n"
        "<open_lines>\n"
        f"{format_open_lines(open_lines)}\n"
        "</open_lines>\n"
        "</answer>"
    )


def format_full_xml_output(
    open_lines: list[tuple[int, int]],
    node_voltages: list[float],
    system_loss: str,
) -> str:
    return (
        "<answer>\n"
        "<open_lines>\n"
        f"{format_open_lines(open_lines)}\n"
        "</open_lines>\n"
        "<node_voltages>\n"
        f"{format_float_list(node_voltages)}\n"
        "</node_voltages>\n"
        "<system_loss>\n"
        f"{system_loss}\n"
        "</system_loss>\n"
        "</answer>"
    )


def format_output(row: dict[str, str], output_mode: str) -> str:
    open_lines = parse_literal(row["updated_open_lines"])
    if output_mode == "full_output":
        node_voltages = parse_literal(row["updated_node_voltages"])
        return format_full_xml_output(
            open_lines=open_lines,
            node_voltages=node_voltages,
            system_loss=row["updated_system_loss"],
        )
    return format_open_lines_xml_output(open_lines)


def split_values(num_samples: int) -> list[str]:
    np.random.seed(42)
    values = (
        ["train"] * (num_samples // 3)
        + ["test"] * (num_samples // 3)
        + ["validation"] * (num_samples // 3)
    )
    remainder = num_samples - len(values)
    values += np.random.choice(["train", "test", "validation"], size=remainder).tolist()
    np.random.shuffle(values)
    return values


def read_unprocessed_rows(unprocessed_dir: Path, bus_systems: list[int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for bus in bus_systems:
        path = unprocessed_dir / f"samples_{bus}bus.csv"
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                row["_source_bus"] = str(bus)
                rows.append(row)
    return rows


def split_values_by_source(unprocessed_rows: list[dict[str, str]]) -> list[str]:
    if not unprocessed_rows or "_source_bus" not in unprocessed_rows[0]:
        return split_values(len(unprocessed_rows))

    values: list[str] = []
    start = 0
    while start < len(unprocessed_rows):
        source_bus = unprocessed_rows[start]["_source_bus"]
        end = start
        while end < len(unprocessed_rows) and unprocessed_rows[end]["_source_bus"] == source_bus:
            end += 1
        values.extend(split_values(end - start))
        start = end
    return values


def build_rows(
    unprocessed_rows: list[dict[str, str]],
    output_mode: str,
) -> list[dict[str, str]]:
    desc = task_description(output_mode)
    processed_rows = []
    source_counts: dict[str, int] = {}
    for global_idx, (row, split) in enumerate(
        zip(unprocessed_rows, split_values_by_source(unprocessed_rows))
    ):
        source_key = row.get("_source_bus", "_all")
        local_idx = source_counts.get(source_key, 0)
        source_counts[source_key] = local_idx + 1
        input_text = format_input(row)
        processed_rows.append(
            {
                "id": str(local_idx if "_source_bus" in row else global_idx),
                "Task Description": desc,
                "input": input_text,
                "prompt": f"{desc}\n{input_text}",
                "output": format_output(row, output_mode),
                "split": split,
            }
        )
    return processed_rows


def write_processed_csv(rows: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "Task Description", "input", "prompt", "output", "split"]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    raw_rows = read_unprocessed_rows(args.unprocessed_dir, args.bus_systems)
    rows = build_rows(raw_rows, args.output_mode)
    write_processed_csv(rows, args.output_csv)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
