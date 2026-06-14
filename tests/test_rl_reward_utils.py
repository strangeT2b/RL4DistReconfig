import unittest

from RL.reward import compute_reward, group_advantages
from utils.metrics_utils import (
    compute_cycles_loss,
    compute_invalid_edges_loss,
    compute_subgraphs_loss,
    graph_penalties_from_open_lines,
    get_output_graph_edges,
    parse_available_lines,
    parse_num_buses,
    parse_open_lines,
)


class RlRewardUtilsTests(unittest.TestCase):
    def test_parses_lines_from_prompt_and_response(self):
        prompt = "Power Distribution Network: Lines=[(1, 2), (2, 3), (3, 1), (3, 4)]"
        response = "Output: Open Lines=[(3, 1)], Node Voltages=[1.0], System Loss=0.12"

        self.assertEqual(parse_available_lines(prompt), [(1, 2), (2, 3), (3, 1), (3, 4)])
        self.assertEqual(parse_num_buses(prompt), 0)
        self.assertEqual(parse_open_lines(response), [(3, 1)])

    def test_reward_prefers_valid_radial_output(self):
        prompt = "Power Distribution Network: Lines=[(1, 2), (2, 3), (3, 1), (3, 4)]"
        valid = "Output: Open Lines=[(3, 1)], Node Voltages=[1.0], System Loss=0.12"
        invalid = "Output: Open Lines=[(9, 10)], Node Voltages=[1.0], System Loss=0.12"
        malformed = "I cannot solve this case."

        valid_reward, valid_parts = compute_reward(prompt, valid)
        invalid_reward, invalid_parts = compute_reward(prompt, invalid)
        malformed_reward, malformed_parts = compute_reward(prompt, malformed)

        self.assertGreater(valid_reward, invalid_reward)
        self.assertGreater(invalid_reward, malformed_reward)
        self.assertEqual(valid_parts["cycles"], 0.0)
        self.assertEqual(valid_parts["subgraphs"], 0.0)
        self.assertEqual(invalid_parts["invalid_edges"], 1.0)
        self.assertEqual(malformed_parts["format_penalty"], 1.0)

    def test_format_penalty_is_reported_but_zero_weight_by_default(self):
        prompt = "Power Distribution Network: Lines=[(1, 2), (2, 3)]"
        malformed = "I cannot solve this case."

        default_reward, default_parts = compute_reward(prompt, malformed)
        shaped_reward, shaped_parts = compute_reward(prompt, malformed, format_penalty_weight=2.0)

        self.assertEqual(default_parts["format_penalty"], 1.0)
        self.assertEqual(shaped_parts["format_penalty"], 1.0)
        self.assertEqual(shaped_reward, default_reward - 2.0)

    def test_group_advantages_are_centered_and_ordered(self):
        advantages = group_advantages([1.0, 2.0, 4.0])

        self.assertAlmostEqual(sum(advantages), 0.0)
        self.assertLess(advantages[0], advantages[1])
        self.assertLess(advantages[1], advantages[2])

    def test_graph_metric_helpers_score_edges(self):
        available_lines = [(1, 2), (2, 3), (3, 1), (3, 4)]
        predicted_open_lines = [(3, 1)]
        graph_edges = get_output_graph_edges(predicted_open_lines, available_lines)

        self.assertEqual(graph_edges, [(1, 2), (2, 3), (3, 4)])
        self.assertEqual(float(compute_invalid_edges_loss([(9, 10)], available_lines)), 1.0)
        self.assertEqual(float(compute_cycles_loss(graph_edges)), 0.0)
        self.assertEqual(float(compute_subgraphs_loss(graph_edges)), 0.0)

    def test_graph_penalties_count_isolated_buses_as_subgraphs(self):
        prompt = (
            "Power Distribution Network: Busses=4, "
            "Lines=[(1, 2), (2, 3), (3, 4), (1, 3), (2, 4)]"
        )
        # Closing only (1,2) and (2,3) leaves bus 4 isolated. An edge-only
        # graph would miss bus 4, so this specifically guards add_nodes_from.
        parts = graph_penalties_from_open_lines(
            prompt,
            [(3, 4), (1, 3), (2, 4)],
        )

        self.assertEqual(parts["invalid_edges"], 0.0)
        self.assertEqual(parts["cycles"], 0.0)
        self.assertGreater(parts["subgraphs"], 0.0)

    def test_graph_penalties_detect_wrong_open_line_count_for_84_bus_shape(self):
        tree_edges = [(i, i + 1) for i in range(1, 84)]
        tie_edges = [(1, 84)] + [(i, i + 20) for i in range(1, 13)]
        all_edges = tree_edges + tie_edges
        prompt = f"Power Distribution Network: Busses=84, Lines={all_edges}"

        parts = graph_penalties_from_open_lines(prompt, all_edges[-15:])

        self.assertEqual(len(all_edges), 96)
        self.assertEqual(parts["invalid_edges"], 0.0)
        self.assertEqual(parts["cycles"], 0.0)
        self.assertGreater(parts["subgraphs"], 0.0)

    def test_graph_penalties_detect_cycle_rank(self):
        prompt = (
            "Power Distribution Network: Busses=4, "
            "Lines=[(1, 2), (2, 3), (3, 4), (4, 1), (2, 4)]"
        )

        valid_tree = graph_penalties_from_open_lines(prompt, [(4, 1), (2, 4)])
        one_cycle = graph_penalties_from_open_lines(prompt, [(2, 4)])
        two_cycles = graph_penalties_from_open_lines(prompt, [(9, 10)])

        self.assertEqual(valid_tree["subgraphs"], 0.0)
        self.assertEqual(valid_tree["cycles"], 0.0)
        self.assertEqual(one_cycle["subgraphs"], 0.0)
        self.assertEqual(one_cycle["cycles"], 1.0)
        self.assertEqual(two_cycles["invalid_edges"], 1.0)
        self.assertEqual(two_cycles["subgraphs"], 0.0)
        self.assertEqual(two_cycles["cycles"], 2.0)


if __name__ == "__main__":
    unittest.main()
