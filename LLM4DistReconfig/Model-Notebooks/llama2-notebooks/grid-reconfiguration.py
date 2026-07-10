import argparse
import re
import ast
import networkx as nx
from huggingface_hub import login
import sys
import os
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from trl import SFTTrainer
import subprocess as sp
from time import perf_counter
import nvidia_smi

sys.path.append(os.path.abspath('/path/to/LLM-Reconfiguration/Dataset-Notebooks/utils'))
from dataset_utils import *
from model_utils import *
from generation_utils import *
from monitoring_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with specific configurations.")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset.')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID to use.')
    parser.add_argument('--output_model', type=str, required=True, help='Output model path.')
    parser.add_argument('--num_train_epochs', type=int, required=True, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size per device.')
    parser.add_argument('--max_new_tokens', type=int, required=True, help='Maximum number of new tokens generated.')
    parser.add_argument('--model_name_hf', type=str, required=True, help='Huggingface model name for upload.')
    parser.add_argument('--tokenizer_name_hf', type=str, required=True, help='Huggingface tokenizer name for upload.')
    parser.add_argument('--custom_loss', type=int, required=True, help='For regular loss use 0. For custom loss with 3 penalty terms use 1.')
    parser.add_argument('--custom_loss_config', type=str, required=True, help='Custom loss configuration.')
    parser.add_argument('--cycles_loss_scaling_factor', type=float, required=True, help='Scaling factor for cycles loss')
    parser.add_argument('--model_for_generation_path', type=str, required=True, help='Model path to use for generation.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print('Training configurations: \n', args)

    access_token = 'hf_token'
    login(token=access_token)

    os.getcwd()

    data_path = args.data_path
    model_id = args.model_id
    output_model = args.output_model


    train_dataset, validation_dataset, test_dataset = prepare_train_data(data_path)
    print(train_dataset)


    model, tokenizer = get_model_and_tokenizer(model_id)

    get_memory_allocated_cached()

    peft_config = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        
    training_arguments = TrainingArguments(
            output_dir=output_model,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=10,
            num_train_epochs=args.num_train_epochs,
            fp16=True,
            report_to="none",
            gradient_checkpointing=True
        )
    
    if args.custom_loss==0:
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=4096
        )
        
    elif args.custom_loss==1:
        class CustomTrainer(SFTTrainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                # Get the model loss
                outputs = model(**inputs)
                loss = outputs.loss
        
                # Detokenize inputs and outputs
                input_text = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
                output_text = self.tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
                
                # Get available lines and predicted lines and use them to get the outputted graph edges
                available_lines = parse_available_lines(input_text[0])
                predicted_lines = parse_open_lines(output_text[0])
                graph_edges = get_output_graph_edges(predicted_lines, available_lines)
                print(predicted_lines)
                
                # Calculate your additional loss terms
                if predicted_lines != []:
                    if "IEL" in args.custom_loss_config:
                        invalid_edges_loss = compute_invalid_edges_loss(predicted_lines, available_lines)/len(predicted_lines)
                    if "CYL" in args.custom_loss_config:
                        cycles_loss = compute_cycles_loss(graph_edges)/len(available_lines)
                    if "SUL" in args.custom_loss_config:    
                        subgraphs_loss = compute_subgraphs_loss(graph_edges)/len(predicted_lines)
                else:
                    if "IEL" in args.custom_loss_config:
                        invalid_edges_loss = 1
                    if "CYL" in args.custom_loss_config:    
                        cycles_loss = 1
                    if "SUL" in args.custom_loss_config:
                        subgraphs_loss = 1
                
                total_loss = loss
                # Add the additional loss terms to the main loss
                if "IEL" in args.custom_loss_config:
                        total_loss += invalid_edges_loss
                if "CYL" in args.custom_loss_config:
                        total_loss += cycles_loss * args.cycles_loss_scaling_factor
                if "SUL" in args.custom_loss_config:    
                        total_loss += subgraphs_loss
                
                
                if "IEL" in args.custom_loss_config:
                        print('Invalid edges loss: ', invalid_edges_loss)
                if "CYL" in args.custom_loss_config:
                        print('Cycles Loss: ', cycles_loss)
                if "SUL" in args.custom_loss_config:    
                        print('Subgraphs loss: ', subgraphs_loss)
                
                print('Total loss: ', total_loss.item())
        
                return (total_loss, outputs) if return_outputs else total_loss
        
        
        trainer = CustomTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            args=training_arguments,
            tokenizer=tokenizer,
            packing=False,
            max_seq_length=4096
        )
        
    

    get_memory_allocated_cached()

    get_cuda_info()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device) 

    get_gpu_memory()

    get_nvidia_smi_info(0)

    trainer.train()
                                                 
    model_path = args.model_for_generation_path
    
    model_with_peft = peft_merge_unload(model_id, model_path)

    print(model)

    model_with_peft.push_to_hub(args.model_name_hf, access_token)
    tokenizer.push_to_hub(args.tokenizer_name_hf, access_token)

    generate_response(user_input=test_dataset['prompt'][0], model=model_with_peft, tokenizer =tokenizer)

if __name__ == "__main__":
    main()
