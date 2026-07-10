import unittest

from utils.prompt_format_utils import format_prompt, format_sft_text, formatted_prompt


class PromptFormatUtilsTests(unittest.TestCase):
    def test_format_prompt_variants(self):
        self.assertEqual(
            format_prompt("P", "legacy"),
            "<|user|>\nP</s>\n<|assistant|>\n",
        )
        self.assertEqual(
            format_prompt("P", "qwen_chat"),
            "<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\n",
        )
        self.assertEqual(
            format_prompt("P", "llama3_chat"),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "P<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    def test_format_sft_text_variants(self):
        self.assertEqual(
            format_sft_text("P", "O", "legacy"),
            "<|user|>\nP</s>\n<|assistant|>\nO</s>",
        )
        self.assertEqual(
            format_sft_text("P", "O", "qwen_chat"),
            "<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\nO<|im_end|>",
        )
        self.assertEqual(
            format_sft_text("P", "O", "llama3_chat"),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "P<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "O<|eot_id|>",
        )

    def test_formatted_prompt_is_legacy_alias(self):
        self.assertEqual(formatted_prompt("P"), format_prompt("P", "legacy"))


if __name__ == "__main__":
    unittest.main()
