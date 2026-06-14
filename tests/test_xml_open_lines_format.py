import unittest

from RL.reward import (
    compute_reward,
    compute_reward_xml_iou_only,
    compute_reward_xml_valid_and_iou,
    compute_reward_xml_valid_only,
)
from utils.metrics_utils import parse_open_lines_xml


def xml(open_lines: str) -> str:
    return (
        "<answer>\n"
        "<open_lines>\n"
        f"{open_lines}\n"
        "</open_lines>\n"
        "</answer>"
    )


class XmlOpenLinesFormatTests(unittest.TestCase):
    def test_parse_xml_open_lines(self):
        self.assertEqual(parse_open_lines_xml(xml("[(1,2),(4,3)]")), [(1, 2), (4, 3)])

    def test_parse_preserves_reversed_edge_spelling(self):
        self.assertEqual(parse_open_lines_xml(xml("[(2,1)]")), [(2, 1)])

    def test_rejects_trailing_text(self):
        self.assertEqual(parse_open_lines_xml(xml("[(1,2)]") + "\nextra"), [])

    def test_rejects_multiple_tags(self):
        doubled = xml("[(1,2)]") + "\n" + xml("[(2,3)]")
        self.assertEqual(parse_open_lines_xml(doubled), [])

    def test_rejects_invalid_edges(self):
        self.assertEqual(parse_open_lines_xml(xml("[(1,2,3)]")), [])
        self.assertEqual(parse_open_lines_xml(xml("[('a',2)]")), [])
        self.assertEqual(parse_open_lines_xml(xml("{'edge': (1,2)}")), [])

    def test_rejects_author_format(self):
        author = "Output: Open Lines=[(1,2)], Node Voltages=[1.0], System Loss=0.1"
        self.assertEqual(parse_open_lines_xml(author), [])

    def test_xml_rewards_match_author_shape(self):
        prompt = "Power Distribution Network: Lines=[(1, 2), (2, 3), (3, 1), (3, 4)]"
        valid = xml("[(3,1)]")
        gt = xml("[(1,3)]")
        invalid = xml("[(9,10)]")

        reward, parts = compute_reward_xml_valid_only(prompt, valid)
        self.assertEqual(reward, 1.0)
        self.assertEqual(parts["is_valid"], 1.0)

        reward, parts = compute_reward_xml_iou_only(prompt, valid, gt)
        self.assertEqual(reward, 12.0)
        self.assertEqual(parts["gt_exact_match"], 1.0)
        self.assertEqual(parts["edge_precision"], 1.0)
        self.assertEqual(parts["edge_recall"], 1.0)

        reward, parts = compute_reward_xml_valid_and_iou(prompt, valid, gt)
        self.assertEqual(reward, 12.0)
        self.assertEqual(parts["iou"], 1.0)

        reward, parts = compute_reward_xml_valid_and_iou(prompt, invalid, invalid)
        self.assertLess(reward, 0.0)
        self.assertEqual(parts["is_valid"], 0.0)

    def test_xml_iou_only_does_not_gate_on_graph_validity(self):
        prompt = "Power Distribution Network: Lines=[(1, 2), (2, 3), (3, 1), (3, 4)]"
        invalid = xml("[(9,10)]")

        reward, parts = compute_reward_xml_iou_only(prompt, invalid, invalid)
        self.assertEqual(reward, 12.0)
        self.assertGreater(parts["invalid_edges"], 0.0)

    def test_xml_iou_reward_uses_current_improvement_and_copy_penalty(self):
        prompt = (
            "Power Distribution Network: Lines=[(1, 2), (2, 3), (3, 1), (3, 4)], "
            "Open Lines=[(3, 1)]\n"
            "Network Variables: NodeVoltages=[1.0], System Loss=1.0"
        )
        copied_current = xml("[(1,3)]")
        better_than_current = xml("[(2,3)]")
        gt = xml("[(2,3)]")

        reward, parts = compute_reward_xml_iou_only(prompt, better_than_current, gt)
        self.assertEqual(reward, 12.0)
        self.assertEqual(parts["iou"], 1.0)
        self.assertEqual(parts["iou_current_gt"], 0.0)
        self.assertEqual(parts["copied_current_open"], 0.0)

        reward, parts = compute_reward_xml_iou_only(prompt, copied_current, gt)
        self.assertEqual(reward, -2.0)
        self.assertEqual(parts["iou"], 0.0)
        self.assertEqual(parts["edge_precision"], 0.0)
        self.assertEqual(parts["edge_recall"], 0.0)
        self.assertEqual(parts["copied_current_open"], 1.0)
        self.assertEqual(parts["copy_penalty"], 2.0)

    def test_author_reward_still_uses_author_format(self):
        prompt = "Power Distribution Network: Lines=[(1, 2), (2, 3), (3, 1), (3, 4)]"
        author = "Output: Open Lines=[(3, 1)], Node Voltages=[1.0], System Loss=0.12"

        reward, parts = compute_reward(prompt, author)
        self.assertGreater(reward, -1.0)
        self.assertEqual(parts["format_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
