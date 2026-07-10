"""Tests for grid simulator — parsing and power flow."""

from __future__ import annotations

import csv
import unittest


def _load_sample(split: str = "train") -> tuple[str, str]:
    with open("Dataset/Processed/train_33_69_84_nodes.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                return row["prompt"], row["output"]
    raise RuntimeError(f"No {split} sample found")


def _has_pandapower() -> bool:
    try:
        import pandapower  # noqa: F401
        import pandapower.networks  # noqa: F401
        net = pandapower.networks.case33bw()
        pandapower.runpp(net, algorithm="bfsw", numba=False)
        return bool(net.converged)
    except Exception:
        return False


class ParsingTests(unittest.TestCase):
    """Test prompt parsing — no pandapower needed."""

    def test_01_parse_33_bus(self):
        from RL.simulator.grid_simulator import parse_grid_from_prompt

        prompt, _ = _load_sample("train")
        params = parse_grid_from_prompt(prompt)
        self.assertEqual(params.busses, 33)
        self.assertEqual(len(params.topology), 37)
        self.assertEqual(len(params.impedances), 37)
        self.assertEqual(len(params.open_lines), 5)
        self.assertEqual(len(params.voltages), 33)
        self.assertEqual(len(params.loads), 33)
        self.assertGreater(params.system_loss, 0)

    def test_02_parse_69_bus(self):
        from RL.simulator.grid_simulator import parse_grid_from_prompt

        prompt, _ = _load_sample("validation")
        params = parse_grid_from_prompt(prompt)
        self.assertIn(params.busses, [33, 69, 84])
        self.assertEqual(len(params.topology), len(params.impedances))

    def test_03_open_lines_are_tuples(self):
        from RL.simulator.grid_simulator import parse_grid_from_prompt

        prompt, _ = _load_sample("train")
        params = parse_grid_from_prompt(prompt)
        self.assertIsInstance(params.open_lines[0], tuple)
        self.assertEqual(len(params.open_lines[0]), 2)

    def test_04_loads_are_complex(self):
        from RL.simulator.grid_simulator import parse_grid_from_prompt

        prompt, _ = _load_sample("train")
        params = parse_grid_from_prompt(prompt)
        self.assertIsInstance(params.loads[0], complex)
        self.assertEqual(params.loads[0], 0j)

    def test_05_extract_output_open_lines(self):
        from RL.simulator.grid_simulator import _re_tuples

        _, output = _load_sample("train")
        open_lines = _re_tuples(r"Open Lines=\[(.*?)\]", output)
        self.assertTrue(len(open_lines) > 0)
        self.assertNotEqual(open_lines, [])
        self.assertIsInstance(open_lines[0], tuple)


class PowerFlowTests(unittest.TestCase):
    """Test pandapower integration — requires functional pandapower."""

    def setUp(self):
        if not _has_pandapower():
            self.skipTest("pandapower not functional (numpy/numba mismatch)")

    def test_01_build_network(self):
        from RL.simulator.grid_simulator import build_network, parse_grid_from_prompt

        prompt, _ = _load_sample("train")
        params = parse_grid_from_prompt(prompt)
        net = build_network(params)
        self.assertEqual(len(net.bus), 33)
        self.assertEqual(len(net.line), 37)
        self.assertEqual(len(net.load), 32)  # bus 0 has 0j load, skipped

    def test_02_base_case_converges(self):
        from RL.simulator.grid_simulator import (
            apply_reconfig, build_network, parse_grid_from_prompt, run_power_flow,
        )

        prompt, _ = _load_sample("train")
        params = parse_grid_from_prompt(prompt)
        net = build_network(params)
        apply_reconfig(net, params.open_lines)
        result = run_power_flow(net)
        self.assertTrue(result["converged"], "Base case should converge")

    def test_03_author_reconfig(self):
        from RL.simulator.grid_simulator import (
            _re_tuples, apply_reconfig, build_network,
            parse_grid_from_prompt, run_power_flow,
        )

        prompt, output = _load_sample("train")
        params = parse_grid_from_prompt(prompt)
        author_open = _re_tuples(r"Open Lines=\[(.*?)\]", output)

        net = build_network(params)
        apply_reconfig(net, author_open)
        result = run_power_flow(net)
        self.assertTrue(result["converged"], "Author reconfig should converge")

    def test_04_evaluate_reconfig(self):
        from RL.simulator.grid_simulator import _re_tuples, evaluate_reconfig

        prompt, output = _load_sample("train")
        author_open = _re_tuples(r"Open Lines=\[(.*?)\]", output)
        result = evaluate_reconfig(prompt, author_open)
        self.assertTrue(result["converged"])
        # improvement_mw may be negative with our approximate model
        self.assertIn("improvement_mw", result)


if __name__ == "__main__":
    unittest.main()
