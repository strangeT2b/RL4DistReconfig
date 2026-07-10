import importlib.util
import os
import sys
from unittest import mock
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "SFT" / "reproduce_author_llama.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("reproduce_author_llama", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReproduceAuthorLlamaTests(unittest.TestCase):
    def test_upstream_utils_path_is_project_local(self):
        module = load_script_module()

        self.assertTrue(module.UPSTREAM_UTILS_DIR.exists())
        self.assertIn("LLM4DistReconfig", str(module.UPSTREAM_UTILS_DIR))
        self.assertNotIn("/path/to", str(module.UPSTREAM_UTILS_DIR))

    def test_ensure_upstream_utils_on_path(self):
        module = load_script_module()
        utils_path = str(module.UPSTREAM_UTILS_DIR)
        old_path = list(sys.path)
        try:
            sys.path = [path for path in sys.path if path != utils_path]
            module.ensure_upstream_utils_on_path()
            self.assertEqual(sys.path[0], utils_path)
        finally:
            sys.path = old_path

    def test_author_custom_loss_arguments_are_required(self):
        module = load_script_module()
        with self.assertRaises(SystemExit):
            module.parse_args(
                [
                    "--data_path",
                    "data.csv",
                    "--model_id",
                    "model",
                    "--output_root",
                    "runs",
                    "--run_name",
                    "out",
                    "--num_train_epochs",
                    "1",
                    "--batch_size",
                    "4",
                ]
            )

        args = module.parse_args(
            [
                "--data_path",
                "data.csv",
                "--model_id",
                "model",
                "--output_root",
                "runs",
                "--run_name",
                "out",
                "--num_train_epochs",
                "1",
                "--batch_size",
                "4",
                "--custom_loss",
                "0",
                "--custom_loss_config",
                "IEL,SUL,CYL",
                "--cycles_loss_scaling_factor",
                "1",
                "--max_new_tokens",
                "1200",
                "--model_name_hf",
                "unused-model",
                "--tokenizer_name_hf",
                "unused-tokenizer",
                "--model_for_generation_path",
                "unused-checkpoint",
            ]
        )
        self.assertEqual(args.custom_loss, 0)
        self.assertEqual(args.custom_loss_config, "IEL,SUL,CYL")
        self.assertEqual(args.cycles_loss_scaling_factor, 1.0)
        self.assertEqual(args.max_new_tokens, 1200)
        self.assertEqual(args.model_name_hf, "unused-model")
        self.assertEqual(args.tokenizer_name_hf, "unused-tokenizer")
        self.assertEqual(args.model_for_generation_path, "unused-checkpoint")
        self.assertEqual(args.save_strategy, "epoch")
        self.assertEqual(args.save_steps, 100)
        self.assertEqual(args.max_steps, -1)
        self.assertEqual(args.gradient_accumulation_steps, 4)

    def test_checkpoint_save_arguments_can_be_set_explicitly(self):
        module = load_script_module()
        args = module.parse_args(
            [
                "--data_path",
                "data.csv",
                "--model_id",
                "model",
                "--output_root",
                "runs",
                "--run_name",
                "out",
                "--num_train_epochs",
                "1",
                "--batch_size",
                "4",
                "--custom_loss",
                "0",
                "--custom_loss_config",
                "IEL,SUL,CYL",
                "--cycles_loss_scaling_factor",
                "1",
                "--max_new_tokens",
                "1200",
                "--model_name_hf",
                "unused-model",
                "--tokenizer_name_hf",
                "unused-tokenizer",
                "--model_for_generation_path",
                "unused-checkpoint",
                "--save_strategy",
                "steps",
                "--save_steps",
                "25",
                "--max_steps",
                "300",
                "--gradient_accumulation_steps",
                "2",
            ]
        )

        self.assertEqual(args.save_strategy, "steps")
        self.assertEqual(args.save_steps, 25)
        self.assertEqual(args.max_steps, 300)
        self.assertEqual(args.gradient_accumulation_steps, 2)

    def test_filter_predicted_lines_keeps_only_author_expected_edge_pairs(self):
        module = load_script_module()

        self.assertEqual(module.filter_predicted_lines([(1, 2), [3, 4]]), [(1, 2), [3, 4]])
        self.assertEqual(module.filter_predicted_lines([1, 2, 3]), [])
        self.assertEqual(module.filter_predicted_lines([(1, 2), 3]), [])

    def test_configure_wandb_sets_optional_environment(self):
        module = load_script_module()
        args = module.parse_args(
            [
                "--data_path",
                "data.csv",
                "--model_id",
                "model",
                "--output_root",
                "runs",
                "--run_name",
                "out",
                "--num_train_epochs",
                "1",
                "--batch_size",
                "4",
                "--max_new_tokens",
                "1200",
                "--model_name_hf",
                "unused-model",
                "--tokenizer_name_hf",
                "unused-tokenizer",
                "--custom_loss",
                "0",
                "--custom_loss_config",
                "IEL,SUL,CYL",
                "--cycles_loss_scaling_factor",
                "1",
                "--model_for_generation_path",
                "unused-checkpoint",
                "--report_to",
                "wandb",
                "--wandb_project",
                "author-repro",
                "--wandb_run_group",
                "llama",
                "--wandb_tags",
                "author, llama ",
            ]
        )
        keys = ["WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_GROUP", "WANDB_TAGS"]
        old_values = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)
            module.configure_wandb(args)
            self.assertEqual(os.environ["WANDB_PROJECT"], "author-repro")
            self.assertNotIn("WANDB_ENTITY", os.environ)
            self.assertEqual(os.environ["WANDB_RUN_GROUP"], "llama")
            self.assertEqual(os.environ["WANDB_TAGS"], "author,llama")
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_main_loads_project_env_before_configuring_wandb(self):
        module = load_script_module()
        env_path = module.REPO_ROOT / ".env"
        with mock.patch.object(module, "ensure_upstream_utils_on_path"):
            with mock.patch.object(module, "load_project_env") as load_env:
                with mock.patch.dict(
                    sys.modules,
                    {
                        "peft": mock.Mock(LoraConfig=mock.Mock()),
                        "dataset_utils": mock.Mock(prepare_train_data=mock.Mock()),
                        "model_utils": mock.Mock(),
                        "transformers": mock.Mock(TrainingArguments=mock.Mock()),
                        "trl": mock.Mock(SFTTrainer=mock.Mock()),
                    },
                ):
                    with mock.patch.object(module, "parse_args", side_effect=RuntimeError("stop")):
                        with self.assertRaises(RuntimeError):
                            module.main()

        load_env.assert_called_once_with(env_path)


if __name__ == "__main__":
    unittest.main()
