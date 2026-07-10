"""Dataset formatting helpers for training scripts."""

from __future__ import annotations

from utils.prompt_format_utils import format_sft_text


def format_dataset_text(dataset, prompt_format: str):
    if prompt_format == "dataset_text":
        return dataset
    if "prompt" not in dataset.column_names or "output" not in dataset.column_names:
        raise ValueError(f"--prompt_format {prompt_format} requires prompt and output columns.")

    def format_row(row):
        return {"text": format_sft_text(row["prompt"], row["output"], prompt_format)}

    return dataset.map(format_row)


def maybe_limit_dataset(dataset, max_samples: int):
    if max_samples > 0:
        return dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def _empty_like(dataset):
    return dataset.select([])


def _message_content(message) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _messages_to_prompt_response(messages: list[dict]) -> tuple[str, str]:
    system_parts = []
    user_parts = []
    assistant_response = ""

    for message in messages:
        role = message.get("role")
        content = _message_content(message)
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_response = content

    if not user_parts or not assistant_response:
        raise ValueError("messages_jsonl rows require at least one user and one assistant message.")

    prompt_parts = [part for part in system_parts + [user_parts[-1]] if part]
    return "\n\n".join(prompt_parts), assistant_response


def _format_reasoning_jsonl(dataset, data_format: str, prompt_format: str):
    if data_format == "prompt_response_jsonl":
        required = {"prompt", "response"}
        missing = sorted(required.difference(dataset.column_names))
        if missing:
            raise ValueError(f"prompt_response_jsonl requires columns {sorted(required)}; missing {missing}.")

        def format_row(row):
            return {"text": format_sft_text(row["prompt"], row["response"], prompt_format)}

        return dataset.map(format_row)

    if data_format == "messages_jsonl":
        if "messages" not in dataset.column_names:
            raise ValueError("messages_jsonl requires a messages column.")

        def format_row(row):
            prompt, response = _messages_to_prompt_response(row["messages"])
            return {"text": format_sft_text(prompt, response, prompt_format)}

        return dataset.map(format_row)

    raise ValueError(f"Unsupported data_format: {data_format}")


def prepare_reasoning_jsonl_data(data_path: str, data_format: str, prompt_format: str):
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=data_path)["train"]
    train_dataset = _format_reasoning_jsonl(dataset, data_format, prompt_format)
    validation_dataset = _empty_like(train_dataset)
    test_dataset = _empty_like(train_dataset)
    return train_dataset, validation_dataset, test_dataset


def prepare_train_data(data_path: str, data_format: str = "csv", prompt_format: str = "legacy"):
    from datasets import Dataset, load_dataset

    if data_format in {"prompt_response_jsonl", "messages_jsonl"}:
        return prepare_reasoning_jsonl_data(data_path, data_format, prompt_format)
    if data_format != "csv":
        raise ValueError(f"Unsupported data_format: {data_format}")

    dataset = load_dataset("csv", data_files=data_path)
    train_dataset = dataset["train"].filter(lambda row: row["split"] == "train")
    validation_dataset = dataset["train"].filter(lambda row: row["split"] == "validation")
    test_dataset = dataset["train"].filter(lambda row: row["split"] == "test")

    train_dataset = train_dataset.remove_columns("split")
    validation_dataset = validation_dataset.remove_columns("split")
    test_dataset = test_dataset.remove_columns("split")

    train_df = train_dataset.to_pandas()
    train_df["text"] = train_df[["prompt", "output"]].apply(
        lambda row: format_sft_text(row["prompt"], row["output"], prompt_format),
        axis=1,
    )
    train_dataset = Dataset.from_pandas(train_df)

    validation_df = validation_dataset.to_pandas()
    validation_df["text"] = validation_df[["prompt", "output"]].apply(
        lambda row: format_sft_text(row["prompt"], row["output"], prompt_format),
        axis=1,
    )
    validation_dataset = Dataset.from_pandas(validation_df)

    test_df = test_dataset.to_pandas()
    test_df["text"] = test_df[["prompt", "output"]].apply(
        lambda row: format_sft_text(row["prompt"], row["output"], prompt_format),
        axis=1,
    )
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, validation_dataset, test_dataset
