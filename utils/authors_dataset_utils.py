"""Author's original data loading logic, preserved for SFT reproduction."""

from datasets import Dataset, load_dataset


def prepare_train_data(data_path):
    dataset = load_dataset('csv', data_files=data_path)
    print(dataset)
    
    # Filter the dataset for the 'train', 'validation', and 'test' splits
    train_dataset = dataset['train'].filter(lambda x: x['split'] == 'train')
    validation_dataset = dataset['train'].filter(lambda x: x['split'] == 'validation')
    test_dataset = dataset['train'].filter(lambda x: x['split'] == 'test')

    # Print the first entry of each dataset to verify
    print(train_dataset[0]['split'])
    print(validation_dataset[0]['split'])
    print(test_dataset[0]['split'])
    
    # Remove the 'split' column as it's no longer needed
    train_dataset = train_dataset.remove_columns('split')
    validation_dataset = validation_dataset.remove_columns('split')
    test_dataset = test_dataset.remove_columns('split')

    train_df = train_dataset.to_pandas()
    train_df["text"] = train_df[["prompt", "output"]].apply(lambda x: "<|user|>\n" + x["prompt"] + "</s>\n<|assistant|>\n" + x["output"] + "</s>", axis=1)
    train_dataset = Dataset.from_pandas(train_df)

    validation_df = validation_dataset.to_pandas()
    validation_df["text"] = validation_df[["prompt", "output"]].apply(lambda x: "<|user|>\n" + x["prompt"] + "</s>\n<|assistant|>\n" + x["output"] + "</s>", axis=1)
    validation_dataset = Dataset.from_pandas(validation_df)
    
    test_df = test_dataset.to_pandas()
    test_df["text"] = test_df[["prompt", "output"]].apply(lambda x: "<|user|>\n" + x["prompt"] + "</s>\n<|assistant|>\n" + x["output"] + "</s>", axis=1)
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, validation_dataset, test_dataset