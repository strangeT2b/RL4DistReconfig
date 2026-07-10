"""Power flow simulator for grid reconfiguration using pandapower.

Parses a prompt (from the dataset), builds a pandapower network, applies a
proposed reconfiguration, runs Newton-Raphson, and returns actual system loss.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any


# Base values per bus size — from author's MATLAB data*.m files.
_V_BASE_KV: dict[int, float] = {33: 12.66, 37: 12.66, 69: 12.66, 84: 11.4, 136: 13.8, 417: 10.0}
_S_BASE_MVA: dict[int, float] = {417: 100.0}  # defaults to 1.0 for unlisted


def _z_base(busses: int) -> float:
    v = _V_BASE_KV.get(busses, 12.66)
    s = _S_BASE_MVA.get(busses, 1.0)
    return v * v / s


# X/R ratios per line position for each bus size.
# From author's MATLAB data*.m files (closed branches) + tie switch data.
# Line positions are 0-indexed, matching the topology order in the dataset.
_IEEE_XR: dict[int, list[float]] = {
    # data33.m — LD (32 closed) + TL (5 tie)
    33: [0.510, 0.509, 0.509, 0.509, 0.863, 3.306, 0.330, 0.718, 0.709, 0.331,
         0.331, 0.787, 1.316, 0.890, 0.730, 1.335, 0.784, 0.954, 0.901, 1.168,
         1.322, 0.683, 0.790, 0.782, 0.509, 0.509, 0.882, 0.871, 0.509, 0.988,
         1.166, 1.555, 1.000, 1.000, 1.000, 1.000, 1.000],
    # data69.m — LD (68 closed) + TL (5 tie), aligned to author topology
    69: [2.400, 2.400, 2.400, 1.171, 0.509, 0.509, 0.510, 0.509, 0.331, 0.331,
         0.330, 0.330, 0.330, 0.331, 0.331, 0.340, 0.331, 0.328, 0.329, 0.331,
         0.331, 0.331, 0.331, 0.330, 2.455, 2.445, 0.331, 0.330, 0.330, 0.336,
         0.331, 0.331, 2.455, 2.445, 1.168, 1.168, 1.167, 1.168, 1.169, 1.166,
         1.261, 1.261, 1.333, 2.471, 2.448, 2.447, 2.446, 0.510, 0.336, 0.509,
         0.509, 0.509, 0.509, 0.336, 0.331, 0.304, 0.509, 0.509, 0.509, 0.509,
         0.304, 0.298, 0.331, 0.340, 1.000, 1.000, 1.000, 1.000, 0.330, 0.509,
         1.000, 0.336, 0.331],
    # data84.m — closed branches (83) + tie switches from Su & Lee 2003 (13)
    84: [3.407, 2.053, 2.053, 2.053, 2.053, 2.053, 3.407, 2.053, 2.053, 2.053,
         2.053, 2.039, 2.053, 2.053, 3.407, 2.053, 2.053, 2.053, 2.053, 2.053,
         2.053, 2.053, 2.053, 2.053, 3.407, 2.053, 2.053, 3.407, 2.053, 2.015,
         2.053, 2.053, 2.053, 2.053, 2.053, 2.053, 2.053, 2.053, 2.053, 2.053,
         2.053, 2.053, 3.407, 2.053, 2.053, 2.053, 3.407, 2.053, 2.053, 2.053,
         2.053, 2.053, 2.053, 2.053, 2.053, 3.407, 2.053, 2.053, 3.407, 2.053,
         2.053, 2.053, 2.053, 3.407, 3.407, 2.053, 3.407, 3.407, 3.407, 3.407,
         3.407, 2.015, 3.407, 3.407, 3.407, 3.407, 3.407, 3.407, 3.407, 2.015,
         2.015, 2.053, 2.053, 2.053, 2.053, 2.053, 2.053, 2.053, 2.015, 2.053,
         2.053, 2.053, 2.053, 2.053, 2.053, 2.053],
    # data417.m — 414 branches, V=10kV, S=100MVA
    417: [0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.844, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.844, 1.180, 0.689, 0.689, 0.689,
          1.018, 0.689, 0.689, 0.689, 1.018, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.689, 1.180, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.844, 0.689,
          0.689, 0.689, 0.844, 0.844, 0.844, 0.844, 0.844, 0.689, 0.844, 0.689,
          0.844, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.689, 0.844, 0.844,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.844, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.689, 0.844, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.844, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.844, 0.689, 0.844, 0.689, 0.844, 0.689, 0.844,
          0.689, 0.844, 0.689, 0.844, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.844,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.844,
          0.689, 0.689, 1.018, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.844, 0.689, 0.844, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.844, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689,
          0.689, 0.844, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.844, 0.689,
          0.689, 0.689, 0.844, 0.689, 0.689, 0.844, 0.689, 0.689, 0.690, 0.689,
          0.689, 1.180, 0.689, 0.689, 0.689, 1.180, 1.018, 0.689, 0.689, 1.018,
          0.689, 0.689, 0.689, 1.018, 0.689, 0.689, 0.690, 0.844, 0.689, 0.689,
          0.689, 1.018, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689,
          0.844, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689,
          0.689, 0.689, 1.018, 0.689, 0.689, 0.689, 1.018, 0.689, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.689, 1.018, 0.689, 1.018, 0.689, 0.689, 0.844, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.689, 1.180, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.844, 0.689, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689,
          0.844, 0.689, 0.689, 0.844, 0.689, 0.689, 0.844, 0.689, 0.689, 0.689,
          0.844, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689, 0.844,
          0.689, 0.689, 0.689, 0.844, 0.689, 0.689, 0.689, 0.689, 0.689, 0.689,
          0.689, 0.844, 0.689, 0.689, 0.689, 0.689, 0.689, 0.844, 0.689, 0.689,
          0.689, 0.689, 0.689, 0.844, 0.689, 0.690],
    # data136.m — LD (135 closed) + TL (21 tie)
    136: [2.308, 2.303, 2.309, 2.308, 2.308, 2.308, 2.308, 0.998, 0.526, 0.998,
          0.526, 0.343, 0.998, 0.526, 0.998, 0.526, 2.308, 2.303, 2.309, 2.308,
          0.526, 2.309, 0.526, 2.308, 2.309, 2.309, 0.998, 0.998, 0.526, 0.526,
          0.998, 0.526, 0.526, 0.526, 0.999, 0.526, 0.526, 0.998, 2.308, 2.308,
          0.343, 2.303, 2.309, 0.526, 2.309, 2.308, 2.308, 2.308, 0.998, 0.998,
          2.309, 2.309, 5.307, 2.308, 2.308, 2.308, 0.526, 0.526, 0.526, 0.526,
          0.526, 2.309, 2.309, 2.309, 2.308, 2.308, 2.308, 2.308, 0.526, 2.308,
          0.526, 0.526, 2.308, 0.343, 2.307, 2.308, 2.308, 2.308, 2.309, 0.998,
          0.999, 0.526, 0.998, 0.526, 2.307, 2.308, 1.299, 2.308, 0.343, 2.308,
          2.308, 2.308, 2.308, 0.998, 0.999, 0.999, 2.308, 2.309, 2.308, 2.308,
          2.308, 0.343, 2.308, 2.308, 2.308, 1.299, 1.299, 0.526, 0.526, 1.299,
          0.526, 0.526, 0.526, 0.526, 0.526, 0.526, 0.526, 2.308, 2.308, 2.308,
          2.307, 2.308, 2.309, 0.343, 2.308, 0.526, 2.309, 2.308, 2.309, 2.308,
          2.309, 2.308, 0.998, 0.998, 0.998,
          2.308, 0.526, 0.999, 0.998, 2.309, 0.999, 0.999, 2.308, 2.309, 2.309,
          0.526, 0.999, 2.308, 2.308, 2.308, 2.308, 0.999, 1.299, 0.526, 2.308,
          0.998],
}


@dataclass
class GridParams:
    busses: int
    topology: list[tuple[int, int]]
    impedances: list[float]
    open_lines: list[tuple[int, int]]
    voltages: list[float]
    system_loss: float
    loads: list[complex]


def parse_grid_from_prompt(prompt_text: str) -> GridParams:
    """Parse all grid fields from a dataset prompt string.

    The prompt has a task-description section (with template placeholders like
    "Open Lines=[List all predicted open lines...]") followed by the actual
    data starting with "Power Distribution Network:".  We target the data
    section to avoid matching the template text.
    """
    # Split off the data section: everything after "Power Distribution Network:"
    data_start = prompt_text.find("Power Distribution Network:")
    if data_start >= 0:
        data_section = prompt_text[data_start:]
    else:
        data_section = prompt_text

    # Within the data section, "Network Variables:" may prefix the second line
    nv_pos = data_section.find("Network Variables:")

    busses = _re_int(r"Busses=(\d+)", data_section)
    topology = _re_tuples(r"Lines=\[(.*?)\]", data_section)
    impedances = _re_floats(r"Line Impedances=\[(.*?)\]", data_section)
    open_lines = _re_tuples(r"Open Lines=\[(.*?)\]", data_section, after="Line Impedances=")

    # Voltages, loss, loads come after "Network Variables:" or after the first line
    if nv_pos >= 0:
        nv_section = data_section[nv_pos:]
    else:
        nv_section = data_section

    voltages = _re_floats(r"NodeVoltages=\[(.*?)\]", nv_section)
    system_loss = _re_float(r"System Loss=([\d.]+)", nv_section)
    loads = _re_complex(r"System Load=\[(.*?)\]", nv_section)

    return GridParams(
        busses=busses,
        topology=topology,
        impedances=impedances,
        open_lines=open_lines,
        voltages=voltages,
        system_loss=system_loss,
        loads=loads,
    )


def build_network(params: GridParams) -> Any:
    """Build a pandapower network from parsed grid parameters.

    Author data: |Z| magnitudes in per-unit on 1 MVA / 12.66 kV base.
    Prompt stores |Z| = sqrt(R²+X²).  We recover R and X using per-line
    X/R ratios from the IEEE standard test cases.
    """
    import pandapower as pp

    xr_list = _IEEE_XR.get(params.busses, [])
    if len(xr_list) != len(params.impedances):
        xr_list = []

    v_base = _V_BASE_KV.get(params.busses, 12.66)
    z_base = _z_base(params.busses)

    net = pp.create_empty_network()

    for i in range(params.busses):
        pp.create_bus(net, vn_kv=v_base, name=str(i),
                      max_vm_pu=1.05, min_vm_pu=0.95)

    v0 = params.voltages[0] if params.voltages else 1.0
    pp.create_ext_grid(net, bus=0, vm_pu=v0, va_degree=0.0)

    for idx, ((a, b), z_pu) in enumerate(zip(params.topology, params.impedances)):
        # Zero-impedance branches → bus-bus switch (fuses two nodes into one)
        if abs(z_pu) < 1e-10:
            pp.create_switch(net, bus=a - 1, element=b - 1, et="b", closed=True)
            continue

        z_mag_ohm = z_pu * z_base
        if xr_list:
            xr = xr_list[idx]
            r_ohm = z_mag_ohm / (1 + xr * xr) ** 0.5
            x_ohm = r_ohm * xr
        else:
            r_ohm = z_mag_ohm / 1.118  # assume X/R=0.5
            x_ohm = r_ohm * 0.5

        pp.create_line_from_parameters(
            net, from_bus=a - 1, to_bus=b - 1,
            length_km=1.0, r_ohm_per_km=r_ohm, x_ohm_per_km=x_ohm,
            c_nf_per_km=0.0, max_i_ka=1.0,
        )

    s_base = _S_BASE_MVA.get(params.busses, 1.0)
    for i, s in enumerate(params.loads):
        if abs(s) > 1e-12:
            pp.create_load(net, bus=i,
                           p_mw=s.real * s_base,
                           q_mvar=s.imag * s_base)

    return net


def apply_reconfig(net: Any, open_lines: list[tuple[int, int]]) -> None:
    """Set lines and switches whose edge is in open_lines to out-of-service."""
    open_set = set(open_lines) | set((b, a) for a, b in open_lines)

    for idx in net.line.index:
        fb = int(net.line.at[idx, "from_bus"]) + 1
        tb = int(net.line.at[idx, "to_bus"]) + 1
        net.line.at[idx, "in_service"] = (fb, tb) not in open_set

    if hasattr(net, "switch") and len(net.switch) > 0:
        for idx in net.switch.index:
            sw = net.switch.iloc[idx]
            if sw["et"] == "b":
                fb = int(sw["bus"]) + 1
                tb = int(sw["element"]) + 1
                net.switch.at[idx, "closed"] = (fb, tb) not in open_set


def run_power_flow(net: Any) -> dict:
    """Run Newton-Raphson and return results dict."""
    import pandapower as pp

    try:
        pp.runpp(net, algorithm="bfsw", numba=False, tolerance_mva=1e-8, max_iteration=100)
        converged = bool(net.converged)
    except Exception:
        converged = False

    if not converged:
        return {"converged": False}

    total_loss = float(net.res_line["pl_mw"].sum())
    voltages = net.res_bus["vm_pu"].to_list()
    v_min = min(voltages)
    v_max = max(voltages)

    return {
        "converged": True,
        "system_loss": total_loss,
        "voltages": voltages,
        "v_min": v_min,
        "v_max": v_max,
        "has_voltage_violation": v_min < 0.95 or v_max > 1.05,
    }


def evaluate_reconfig(prompt_text: str, proposed_open_lines: list[tuple[int, int]]) -> dict:
    """Full evaluation: parse prompt, build network, apply reconfig, run power flow."""
    params = parse_grid_from_prompt(prompt_text)
    net = build_network(params)
    apply_reconfig(net, proposed_open_lines)
    result = run_power_flow(net)
    result["original_loss_mw"] = result.get("system_loss", None)  # placeholder, will be overwritten
    if result["converged"]:
        # improvement = original_loss_MW - new_loss_MW (positive = better)
        # We compute original by running base case with original open lines
        net_orig = build_network(params)
        apply_reconfig(net_orig, params.open_lines)
        import pandapower as pp
        pp.runpp(net_orig, algorithm="bfsw", numba=False, tolerance_mva=1e-8, max_iteration=100)
        if net_orig.converged:
            result["original_loss_mw"] = float(net_orig.res_line["pl_mw"].sum())
            result["improvement_mw"] = result["original_loss_mw"] - result["system_loss"]
    return result


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _re_int(pattern: str, text: str) -> int:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"Pattern not found: {pattern}")
    return int(m.group(1))


def _re_float(pattern: str, text: str) -> float:
    m = re.search(pattern, text)
    if not m:
        raise ValueError(f"Pattern not found: {pattern}")
    return float(m.group(1))


def _re_tuples(pattern: str, text: str, after: str = "") -> list[tuple[int, int]]:
    if after:
        pos = text.find(after)
        if pos >= 0:
            text = text[pos + len(after):]
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return []
    content = "[" + m.group(1) + "]"
    try:
        raw = ast.literal_eval(content)
        return [tuple(pair) for pair in raw]
    except (ValueError, SyntaxError):
        return []


def _re_floats(pattern: str, text: str) -> list[float]:
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return []
    content = "[" + m.group(1) + "]"
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return []


def _re_complex(pattern: str, text: str) -> list[complex]:
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return []
    content = "[" + m.group(1) + "]"
    try:
        return ast.literal_eval(content)
    except (ValueError, SyntaxError):
        return []
