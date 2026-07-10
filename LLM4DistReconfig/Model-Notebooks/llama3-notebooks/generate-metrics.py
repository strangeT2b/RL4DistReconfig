import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
import os
import re
import sys
import argparse

sys.path.append(os.path.abspath('/path/to/LLM-Reconfiguration/Dataset-Notebooks/utils'))

from dataset_utils import *
from model_utils import *
from generation_utils import *
from metrics_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with specific configurations.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use.')
    parser.add_argument('--max_new_tokens', type=int, required=True, help='Maximum number of new tokens generated.')
    parser.add_argument('--model_for_generation_path', type=str, required=True, help='Model path to use for generation.')
    parser.add_argument('--filename_txt', type=str, required=True, help='File path and name to save the metrics as txt.')
    parser.add_argument('--filename_csv', type=str, required=True, help='File path and name to save the metrics as csv.')
    parser.add_argument('--num_samples', type=int, required=True, help='Number of samples generated.')
    parser.add_argument('--untrained', type=str, default='False', help='Is the model untrained.')
    return parser.parse_args()

def main():
    
    args = parse_args()
    
    print('Training configurations: \n', args)
    
    data_path = args.data_path
    model_id = args.model_id
    untrained = args.untrained

    train_dataset, validation_dataset, test_dataset = prepare_train_data(data_path)

    model = get_model(model_id)
    tokenizer = get_tokenizer(model_id)
    
    if untrained == 'False':
        peft_config = LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )
        
        model_path = args.model_for_generation_path
        
        model = peft_merge_unload(model_id, model_path)

    from functools import partial

    # Create a partial function with fixed model and tokenizer
    partial_generate_response = partial(generate_response, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)

    generate_metrics(test_dataset, partial_generate_response, num_samples=args.num_samples, filename_txt=args.filename_txt, filename_csv = args.filename_csv)
    
    
if __name__ == "__main__":
    main()
