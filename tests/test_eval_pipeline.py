"""Smoke test for evaluation pipeline (no GPU needed)."""

from __future__ import annotations

import os
import tempfile
import unittest


class MockHFDataset:
    """Minimal HuggingFace dataset-like object for testing eval functions."""

    def __init__(self, samples: list[dict]):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, key):
        if isinstance(key, str):
            # Column access: dataset["prompt"] -> list of values
            return MockHFColumn([s[key] for s in self._samples])
        # Row access: dataset[0] -> dict
        return self._samples[key]


class MockHFColumn:
    """Minimal list-like for dataset column access."""

    def __init__(self, values):
        self._values = values

    def __getitem__(self, idx):
        return self._values[idx]

    def __len__(self):
        return len(self._values)


class EvalPipelineSmokeTest(unittest.TestCase):
    """Test that the eval pipeline imports and basic functions work."""

    def test_01_all_imports_work(self):
        """All eval functions should import without errors."""
        from utils.generation_utils import (
            extract_output_data,
        )
        from utils.metrics_utils import (
            compute_cycles_loss,
            compute_invalid_edges_loss,
            compute_subgraphs_loss,
            extract_metrics,
            get_number_of_nodes,
            get_output_graph_edges,
            graph_penalties,
            parse_available_lines,
            parse_correct_output,
            parse_open_lines,
            prep_csv,
            write_to_csv,
            write_to_txt,
        )
        from utils.model_utils import (
            get_model,
            get_tokenizer,
            peft_merge_unload,
        )
        self.assertTrue(True)

    def test_03_extract_output_data_parses_valid_response(self):
        """extract_output_data should parse a well-formed response."""
        from utils.generation_utils import extract_output_data

        response = "Open Lines=[(1,2),(3,4)], Node Voltages=[0.95,0.87], System Loss=0.123"
        result = extract_output_data(response)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["Open Lines"], [(1, 2), (3, 4)])
        self.assertEqual(result["Node Voltages"], [0.95, 0.87])
        self.assertEqual(result["System Loss"], 0.123)

    def test_04_extract_output_data_handles_bad_response(self):
        """extract_output_data should return string for malformed response."""
        from utils.generation_utils import extract_output_data

        result = extract_output_data("garbage output")
        self.assertEqual(result, "No output data found in the response.")

    def test_05_get_number_of_nodes(self):
        """get_number_of_nodes should return max node number."""
        from utils.metrics_utils import get_number_of_nodes

        lines = [(1, 2), (3, 4), (2, 5)]
        self.assertEqual(get_number_of_nodes(lines), 5)
        self.assertEqual(get_number_of_nodes([]), 0)

    def test_07_csv_writer_roundtrip(self):
        """CSV writer should write dict rows in the order declared by `columns`."""
        from utils.metrics_utils import prep_csv, write_to_csv
        import csv as _csv

        columns = ["dataset_index", "prompt", "gen_open_lines", "is_valid"]
        rows = [
            {"dataset_index": 0, "prompt": "p0", "gen_open_lines": [(1, 2)], "is_valid": 1},
            {"dataset_index": 1, "prompt": "p1", "is_valid": 0},  # partial row
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "x.csv")
            prep_csv(path, columns)
            write_to_csv(path, rows, columns)
            with open(path) as f:
                reader = _csv.DictReader(f)
                read = list(reader)
        self.assertEqual(read[0]["dataset_index"], "0")
        self.assertEqual(read[0]["is_valid"], "1")
        self.assertEqual(read[1]["gen_open_lines"], "")  # missing key → empty

    def test_09_compute_gt_match_undirected(self):
        """compute_gt_match should treat edges as undirected and dedupe."""
        from utils.metrics_utils import compute_gt_match

        # Identical sets, edges flipped → exact match, IoU=1.
        exact, iou = compute_gt_match([(1, 2), (3, 4)], [(2, 1), (4, 3)])
        self.assertEqual(exact, 1.0)
        self.assertEqual(iou, 1.0)

        # Partial overlap: 1 shared / 3 union → IoU = 1/3, exact = 0.
        exact, iou = compute_gt_match([(1, 2), (3, 4)], [(1, 2), (5, 6)])
        self.assertEqual(exact, 0.0)
        self.assertAlmostEqual(iou, 1 / 3)

        # Empty both → vacuously equal.
        exact, iou = compute_gt_match([], [])
        self.assertEqual(exact, 1.0)
        self.assertEqual(iou, 1.0)

        # Malformed entries dropped silently.
        exact, iou = compute_gt_match([(1, 2), (1, 2, 3), "garbage"], [(1, 2)])
        self.assertEqual(exact, 1.0)
        self.assertEqual(iou, 1.0)

    def test_08_graph_penalties_reuse_across_sft_rl(self):
        """graph_penalties works identically for SFT and RL use cases."""
        from utils.metrics_utils import graph_penalties

        prompt = "Lines=[(1,2),(2,3),(3,1)]"
        good_response = "Open Lines=[(1,2)], Node Voltages=[1.0,0.9,0.8], System Loss=0.05"
        bad_response = "Open Lines=[(1,99)], Node Voltages=[1.0], System Loss=0.05"

        good_result = graph_penalties(prompt, good_response)
        bad_result = graph_penalties(prompt, bad_response)

        self.assertIn("invalid_edges", good_result)
        self.assertIn("cycles", good_result)
        self.assertIn("subgraphs", good_result)
        # Bad response should score worse
        self.assertGreater(bad_result["invalid_edges"], good_result["invalid_edges"])


if __name__ == "__main__":
    unittest.main()
