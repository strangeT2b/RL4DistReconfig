import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
import re
import ast
import networkx as nx


def get_model_and_tokenizer(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model, tokenizer
    
def get_model(model_id):
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype="float16", bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    
    model.config.use_cache=False
    model.config.pretraining_tp=1
    return model
    
def get_tokenizer(model_id):
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer
    
def parse_open_lines(output_string):
    """
    Parses the open lines from the output string.

    Parameters:
    - output_string: The string containing the open lines information.

    Returns:
    A list of tuples representing the open lines.
    """
    # Regular expression to find the "Open Lines" part
    open_lines_pattern = r'Open Lines=\[(.*?)\]'

    # Search for the pattern in the string
    match = re.search(open_lines_pattern, output_string)

    if match:
        open_lines_str = match.group(1)
        
        # Remove any extraneous characters and ensure proper format
        open_lines_str = re.sub(r'[^0-9,() ]', '', open_lines_str)  # Remove invalid characters
        open_lines_str = re.sub(r'\s+', '', open_lines_str)  # Remove extra whitespace
        open_lines_str = open_lines_str.strip(',')  # Remove trailing commas

        # Ensure that open_lines_str is properly formatted for evaluation
        try:
            open_lines = ast.literal_eval(f'[{open_lines_str}]')
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing open lines: {e}")
            return []

        return open_lines
    else:
        return []
        
        
def parse_available_lines(output_string):
        # Regular expression to find the "Lines" part in the prompt which represents the available lines
        open_lines_pattern = r'Lines=\[(.*?)\]'
        
        # Search for the pattern in the string
        match = re.search(open_lines_pattern, output_string)
        
        if match:
            available_lines_str = match.group(1)
            # Convert the string representation of the list to an actual list
            available_lines = ast.literal_eval(f'[{available_lines_str}]')
            return available_lines
        else:
            return []
            
            
def get_output_graph_edges(predicted_lines, available_lines):
    predicted_lines_reverse = reverseTuple(predicted_lines)
    predicted_lines.extend(predicted_lines_reverse)
    return [line for line in available_lines if line not in predicted_lines]            
            
            
            
def compute_invalid_edges_loss(predicted_lines, available_lines):
        # Calculate the loss for invalid edges
        invalid_edges_loss = torch.tensor(0.0)
        for line in predicted_lines:
            if line not in available_lines:
                invalid_edges_loss += 1.0  # Adjust the penalty as needed
        return invalid_edges_loss
        
def compute_cycles_loss(predicted_lines):
    # Calculate the loss for cycles
    G = nx.Graph()
    G.add_edges_from(predicted_lines)
    cycles_loss = torch.tensor(0.0)
    try:
        cycles = list(nx.find_cycle(G, orientation="ignore"))
        cycles_loss += len(cycles)  # Adjust the penalty as needed
    except nx.NetworkXNoCycle:
        pass
    return cycles_loss
    
    
def compute_subgraphs_loss(predicted_lines):
    # Calculate the loss for subgraphs
    G = nx.Graph()
    G.add_edges_from(predicted_lines)
    subgraphs_loss = torch.tensor(0.0)
    print('Number of connected components: ', nx.number_connected_components(G)) #remove when training properly
    if nx.number_connected_components(G) > 1:
        subgraphs_loss += nx.number_connected_components(G) - 1  # Adjust the penalty as needed
    return subgraphs_loss    
    
def reverseTuple(lstOfTuple):
    return [tup[::-1] for tup in lstOfTuple]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            