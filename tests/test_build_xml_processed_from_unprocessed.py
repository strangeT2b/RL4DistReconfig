import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "Dataset"
    / "build_xml_processed_from_unprocessed.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_xml_processed_from_unprocessed", SCRIPT_PATH
)
build_processed = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_processed)


class BuildXmlProcessedFromUnprocessedTests(unittest.TestCase):
    def test_build_xml_rows_from_unprocessed_has_no_legacy_output_instruction(self):
        raw_rows = [
            {
                "buses": "3",
                "lines": "[(1, 2), (2, 3), (1, 3)]",
                "line_impedances": "[0.1, 0.2, 0.3]",
                "existing_connectivitty": "[]",
                "existing_open_lines": "[(1, 3)]",
                "existing_node_voltages": "[1.0, 0.99, 0.98]",
                "existing_system_loss": "1.23",
                "system_load": "[0j, (0.1+0.01j), (0.2+0.02j)]",
                "updated_connectivity": "[]",
                "updated_open_lines": "[(2, 3)]",
                "updated_node_voltages": "[1.0, 0.98, 0.97]",
                "updated_system_loss": "1.0",
            }
        ]

        rows = build_processed.build_rows(raw_rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "0")
        self.assertIn("Return only the following XML format", rows[0]["Task Description"])
        self.assertIn("Return only the following XML format", rows[0]["prompt"])
        self.assertIn("Power Distribution Network: Busses=3", rows[0]["prompt"])
        self.assertNotIn("The output format should be strictly", rows[0]["prompt"])
        self.assertNotIn("Node Voltages=[List the updated node voltages", rows[0]["prompt"])
        self.assertNotIn("System Loss=predicted system loss", rows[0]["prompt"])
        self.assertEqual(
            rows[0]["output"],
            "<answer>\n<open_lines>\n[(2,3)]\n</open_lines>\n</answer>",
        )

    def test_convert_unprocessed_file_to_processed_csv(self):
        fieldnames = [
            "buses",
            "lines",
            "line_impedances",
            "existing_connectivitty",
            "existing_open_lines",
            "existing_node_voltages",
            "existing_system_loss",
            "system_load",
            "updated_connectivity",
            "updated_open_lines",
            "updated_node_voltages",
            "updated_system_loss",
        ]
        row = {
            "buses": "3",
            "lines": "[(1, 2), (2, 3), (1, 3)]",
            "line_impedances": "[0.1, 0.2, 0.3]",
            "existing_connectivitty": "[]",
            "existing_open_lines": "[(1, 3)]",
            "existing_node_voltages": "[1.0, 0.99, 0.98]",
            "existing_system_loss": "1.23",
            "system_load": "[0j, (0.1+0.01j), (0.2+0.02j)]",
            "updated_connectivity": "[]",
            "updated_open_lines": "[(2, 3)]",
            "updated_node_voltages": "[1.0, 0.98, 0.97]",
            "updated_system_loss": "1.0",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = Path(tmp_dir) / "raw"
            input_dir.mkdir()
            input_path = input_dir / "samples_3bus.csv"
            output_path = Path(tmp_dir) / "processed.csv"
            with input_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)

            raw_rows = build_processed.read_unprocessed_rows(input_dir, [3])
            rows = build_processed.build_rows(raw_rows)
            build_processed.write_processed_csv(rows, output_path)

            with output_path.open(newline="", encoding="utf-8") as file:
                converted = list(csv.DictReader(file))

        self.assertEqual(list(converted[0].keys()), [
            "id",
            "Task Description",
            "input",
            "prompt",
            "output",
            "split",
        ])
        self.assertIn("<open_lines>", converted[0]["output"])
        self.assertNotIn("The output format should be strictly", converted[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
