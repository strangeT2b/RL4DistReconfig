import tempfile
import unittest
import importlib.util
from pathlib import Path

from utils.dataset_utils import prepare_train_data


@unittest.skipIf(importlib.util.find_spec("datasets") is None, "datasets is required")
class DatasetUtilsTests(unittest.TestCase):
    def test_prepare_train_data_splits_csv_and_adds_text(self):
        rows = [
            "split,prompt,output",
            "train,train prompt,train output",
            "validation,validation prompt,validation output",
            "test,test prompt,test output",
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "toy.csv"
            data_path.write_text("\n".join(rows), encoding="utf-8")

            train_dataset, validation_dataset, test_dataset = prepare_train_data(str(data_path))

        self.assertEqual(len(train_dataset), 1)
        self.assertEqual(len(validation_dataset), 1)
        self.assertEqual(len(test_dataset), 1)
        self.assertNotIn("split", train_dataset.column_names)
        self.assertIn("text", train_dataset.column_names)
        self.assertIn("train prompt", train_dataset[0]["text"])
        self.assertIn("train output", train_dataset[0]["text"])


if __name__ == "__main__":
    unittest.main()
