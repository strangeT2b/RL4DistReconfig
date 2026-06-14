"""Dataset formatting helpers for training scripts."""

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


def prepare_train_data(data_path: str):
    from datasets import Dataset, load_dataset

    dataset = load_dataset("csv", data_files=data_path)
    train_dataset = dataset["train"].filter(lambda row: row["split"] == "train")
    validation_dataset = dataset["train"].filter(lambda row: row["split"] == "validation")
    test_dataset = dataset["train"].filter(lambda row: row["split"] == "test")

    train_dataset = train_dataset.remove_columns("split")
    validation_dataset = validation_dataset.remove_columns("split")
    test_dataset = test_dataset.remove_columns("split")

    train_df = train_dataset.to_pandas()
    train_df["text"] = train_df[["prompt", "output"]].apply(
        lambda row: format_sft_text(row["prompt"], row["output"], "legacy"),
        axis=1,
    )
    train_dataset = Dataset.from_pandas(train_df)

    validation_df = validation_dataset.to_pandas()
    validation_df["text"] = validation_df[["prompt", "output"]].apply(
        lambda row: format_sft_text(row["prompt"], row["output"], "legacy"),
        axis=1,
    )
    validation_dataset = Dataset.from_pandas(validation_df)

    test_df = test_dataset.to_pandas()
    test_df["text"] = test_df[["prompt", "output"]].apply(
        lambda row: format_sft_text(row["prompt"], row["output"], "legacy"),
        axis=1,
    )
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, validation_dataset, test_dataset
