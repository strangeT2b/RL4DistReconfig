import argparse
import os
import tempfile
import unittest
from pathlib import Path

from utils.config_utils import configure_wandb, load_project_env, report_to_value, uses_wandb


class ConfigUtilsTests(unittest.TestCase):
    def test_report_to_value_parses_none_and_lists(self):
        self.assertEqual(report_to_value("none"), "none")
        self.assertEqual(report_to_value(""), "none")
        self.assertEqual(report_to_value("wandb,tensorboard"), ["wandb", "tensorboard"])

    def test_configure_wandb_sets_optional_environment_values(self):
        args = argparse.Namespace(
            report_to="wandb",
            wandb_project="grid-project",
            wandb_entity="grid-team",
            wandb_run_group="rl",
            wandb_tags="rl, reward ",
        )
        old_values = {key: os.environ.get(key) for key in ["WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_GROUP", "WANDB_TAGS"]}
        try:
            for key in old_values:
                os.environ.pop(key, None)

            configure_wandb(args)

            self.assertTrue(uses_wandb(args.report_to))
            self.assertEqual(os.environ["WANDB_PROJECT"], "grid-project")
            self.assertEqual(os.environ["WANDB_ENTITY"], "grid-team")
            self.assertEqual(os.environ["WANDB_RUN_GROUP"], "rl")
            self.assertEqual(os.environ["WANDB_TAGS"], "rl,reward")
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_load_project_env_overrides_existing_values(self):
        keys = ["WANDB_API_KEY", "WANDB_ENTITY"]
        old_values = {key: os.environ.get(key) for key in keys}
        try:
            os.environ["WANDB_API_KEY"] = "old-key"
            os.environ["WANDB_ENTITY"] = "old-team"
            with tempfile.TemporaryDirectory() as tmpdir:
                env_path = Path(tmpdir) / ".env"
                env_path.write_text(
                    "WANDB_API_KEY=new-key\nWANDB_ENTITY=new-team\n",
                    encoding="utf-8",
                )

                load_project_env(env_path)

            self.assertEqual(os.environ["WANDB_API_KEY"], "new-key")
            self.assertEqual(os.environ["WANDB_ENTITY"], "new-team")
        finally:
            for key, value in old_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
