from datasets import load_dataset, Dataset
import ast
import pickle
import numpy as np
import pandas as pd
import glob as glob

def clean_value(value):
    # Function to remove square brackets and convert string representation of lists to actual lists
    
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def modify_and_save(file_path, set_value, df_column, output_file_path):
    file_path = file_path
    df = pd.read_csv(file_path)
    specific_value = set_value
    df[df_column] = specific_value
    output_file_path = output_file_path
    df.to_csv(output_file_path, index=False)
        
def convert_to_dict(data):
    # Convert each row into a dictionary
    data_dict_list = []
    for _, row in data.iterrows():
        entry = {column: clean_value(value) for column, value in row.items()}
        data_dict_list.append(entry)
    return data_dict_list

def dict_to_pickle(data_dict_list, output_file_path):
    # Save the list of dictionaries to a Pickle file
    file_path = output_file_path
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(data_dict_list, pickle_file)
    print(f"Data successfully saved to {output_file_path}")

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
    

def create_inputs(data):
    # Function to create text inputs
    inputs = []
    for entry in data:
        input = (
            f"Power Distribution Network: Busses={entry['buses']}, Lines={entry['lines']}, "
            f"Line Impedances={entry['line_impedances']}, Open Lines={entry['existing_open_lines']}\n"
            f"Network Variables: NodeVoltages={entry['existing_node_voltages']}, "
            f"System Loss={entry['existing_system_loss']}, System Load={entry['system_load']}\n"
        )
        inputs.append(input)
    return inputs


def create_outputs(data):
    # Function to create text outputs
    outputs = []
    for entry in data:
        output = (
            f"Output: Open Lines={entry['updated_open_lines']}, "
            f"Node Voltages={entry['updated_node_voltages']}, System Loss={entry['updated_system_loss']}\n"
        )
        outputs.append(output)
    return outputs
    
def save_to_txt(file_path, entries):
    with open(file_path, 'w') as f:
        for entry in entries:
            f.write(entry + '<end>')


def load_entries(file_path):
    # Function to load prompts from the text file
    with open(file_path, 'r') as f:
        entries = f.read().strip().split('<end>')

    entries.pop()
    return entries
    
def create_task_descriptions(task_description, inputs):
    task_descriptions = [task_description] * len(inputs)
    return task_descriptions
    
def create_prompts(task_descriptions, inputs):
    prompts = [desc + "\n" + inp for desc, inp in zip(task_descriptions, inputs)]
    return prompts
    
def create_train_df(task_descriptions, inputs, prompts, outputs):
    train_data = {'Task Description': task_descriptions, 'input': inputs, 'prompt': prompts,'output': outputs}
    train_df = pd.DataFrame(train_data)
    return train_df
    
def generate_random_split(train_df):
    num_samples = len(train_df)

    # Generate random splits
    np.random.seed(42)  # For reproducibility
    split_values = ['train'] * (num_samples // 3) + ['test'] * (num_samples // 3) + ['validation'] * (num_samples // 3)
    
    # If there are remaining samples, assign them randomly
    remainder = num_samples - len(split_values)
    split_values += np.random.choice(['train', 'test', 'validation'], size=remainder).tolist()
    
    # Shuffle the split assignments
    np.random.shuffle(split_values)
    
    # Add the split column to the DataFrame
    train_df['split'] = split_values
    
def combine_csv_files(file_paths, output_file_path):
    """
    Combines multiple CSV files into a single DataFrame and resets the index.

    Parameters:
    - file_paths: List of file paths to the CSV files to be combined.
    - output_file_path: Path where the combined CSV file will be saved.
    """
    # Initialize an empty list to store individual DataFrames
    data_frames = []

    # Read each file and append the DataFrame to the list
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data_frames.append(df)

    # Concatenate all DataFrames in the list into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Reset the index to ensure unique indices
    combined_df.reset_index(drop=True, inplace=True)

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_file_path, index=False)
    print(f"Combined CSV saved to {output_file_path}")

def formatted_train(input,response)->str:
    return f"<|user|>\n{input}</s>\n<|assistant|>\n{response}</s>"
    
    
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