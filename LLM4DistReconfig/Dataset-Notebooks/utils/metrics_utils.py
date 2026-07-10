import torch
import sys
import os
import csv

sys.path.append(os.path.abspath('/scratch/pc2442/LLM-Reconfiguration/Dataset-Notebooks/utils'))

from dataset_utils import *
from model_utils import *
from generation_utils import *



def generate_metrics(dataset, response_function, num_samples=10, filename_txt="metrics.txt", filename_csv="metrics.csv"):
    """
    Generates responses and then calculates the cycles, invalid edges and the subgraphs
    from the predicted lines. The function also keeps track of the inference time. 
    It returns the avg cycles, invalid edges and subgraphs per response.
    
    Parameters:
    - dataset: The dataset containing the prompts and correct outputs.
    - response_function: The function to generate responses.
    - num_samples: The number of random samples to generate.
    - filename: The name of the file where the responses will be saved.
    """
    # Ensure dataset has enough entries
    if num_samples > len(dataset):
        raise ValueError("The number of samples requested exceeds the dataset size.")
    
    # Generate random indices
    if num_samples == -1:
        indices = torch.randint(0, len(dataset), (len(dataset),))
    else:
        indices = torch.randint(0, len(dataset), (num_samples,))
        
    prep_csv(filename_csv)

    total_cycles_loss = 0
    total_invalid_edges_loss = 0
    total_subgraphs_loss = 0
    no_response_count = 0
    total_inference_time = 0
    
    # Initialize lists to accumulate data
    all_num_nodes = []
    all_available_lines = []
    all_generated_open_lines = []
    all_generated_node_voltages = []
    all_system_loss = []
    all_correct_open_lines = []
    all_correct_generated_lines = []
    all_correct_system_loss = []
    
    
    for index in indices:
        # Extract the prompt and correct output using the random index
        prompt = dataset['prompt'][index]
        correct_output = dataset['output'][index]
        
        # Generate the response
        response, output_time = response_function(user_input=prompt)
        total_inference_time += output_time

        reformatted_response = extract_output_data(response)

        

        if reformatted_response != "No output data found in the response.":
            
            predicted_lines = parse_open_lines(response)
            available_lines = parse_available_lines(response)
            graph_edges = get_output_graph_edges(predicted_lines, available_lines)
            
            total_cycles_loss += compute_cycles_loss(graph_edges)
            total_invalid_edges_loss += compute_invalid_edges_loss(predicted_lines, available_lines)
            total_subgraphs_loss += compute_subgraphs_loss(graph_edges)
            
            # Extract metrics from the response
            num_nodes , generated_open_lines, generated_node_voltages, system_loss = extract_metrics(available_lines, reformatted_response)
            
            # Parse the correct output for comparison
            correct_open_lines, correct_generated_lines, correct_system_loss = parse_correct_output(correct_output)
            
            # Accumulate the data in lists
            all_num_nodes.append(num_nodes)
            all_available_lines.append(available_lines)
            all_generated_open_lines.append(generated_open_lines)
            all_generated_node_voltages.append(generated_node_voltages)
            all_system_loss.append(system_loss)
            all_correct_open_lines.append(correct_open_lines)
            all_correct_generated_lines.append(correct_generated_lines)
            all_correct_system_loss.append(correct_system_loss)
            
            
 
        else:
            no_response_count += 1
    
    write_to_csv(all_num_nodes, all_available_lines, all_generated_open_lines, all_generated_node_voltages, all_system_loss, all_correct_open_lines, all_correct_generated_lines, all_correct_system_loss, filename_csv)
    if num_samples - no_response_count > 0:
        avg_cycles_loss = total_cycles_loss / (num_samples - no_response_count)
    else:
        avg_cycles_loss = 0  
    # avg_cycles_loss = total_cycles_loss/(num_samples - no_response_count)
    avg_invalid_edges_loss = total_invalid_edges_loss/(num_samples - no_response_count)
    avg_subgraphs_loss = total_subgraphs_loss/(num_samples - no_response_count)
    avg_inference_time = total_inference_time/(num_samples - no_response_count)
    
    write_to_txt(filename_txt, num_samples, avg_cycles_loss, avg_invalid_edges_loss, avg_subgraphs_loss, no_response_count, avg_inference_time)


def extract_metrics(available_lines, reformatted_response):
    num_nodes = get_number_of_nodes(available_lines)
    generated_open_lines = reformatted_response['Open Lines']
    generated_node_voltages = reformatted_response['Node Voltages']
    system_loss = reformatted_response['System Loss']
    
    return num_nodes, generated_open_lines, generated_node_voltages, system_loss
    
def parse_correct_output(correct_output):
    correct_data = extract_output_data(correct_output)
    correct_open_lines = correct_data['Open Lines']
    correct_generated_lines = correct_data['Node Voltages']
    correct_system_loss = correct_data['System Loss']
    
    return correct_open_lines, correct_generated_lines, correct_system_loss
            
def prep_csv(filename):
    # Prepare to write to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow([
            "# of nodes", 
            "available lines", 
            "generated open lines", 
            "generated node voltages", 
            "system loss", 
            "correct open lines", 
            "correct generated lines", 
            "correct system loss"
        ])
    
            
def write_to_csv(num_nodes, available_lines, generated_open_lines, generated_node_voltages, system_loss, correct_open_lines, correct_generated_lines, correct_system_loss, filename):
    # Write the row to the CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the accumulated rows
        writer.writerows(zip(
            num_nodes,
            available_lines,
            generated_open_lines,
            generated_node_voltages,
            system_loss,
            correct_open_lines,
            correct_generated_lines,
            correct_system_loss
        ))
        
def write_to_txt(filename, num_samples, avg_cycles_loss, avg_invalid_edges_loss, avg_subgraphs_loss, no_response_count, avg_inference_time):
    # Open the file to write the responses            
    with open(filename, "w") as file:
            # Write the prompt, response, and correct output to the file with clear separation
            file.write("---- Evaluation Metrics ----\n")
            file.write(f"Number of Samples:\n {num_samples}\n")
            file.write(f"Average Cycles:\n {avg_cycles_loss}\n")
            file.write(f"Average Invalid Edges:\n {avg_invalid_edges_loss}\n")
            file.write(f"Average Subgraphs:\n {avg_subgraphs_loss}\n")
            file.write(f"Responses with improper output:\n {no_response_count}\n")
            file.write(f"Inference Time: {avg_inference_time}\n")  
            
def get_number_of_nodes(available_lines):
    """
    Returns the number of nodes by finding the largest number in the list of tuples.
    
    Parameters:
    - available_lines: A list of tuples representing the available lines.
    
    Returns:
    The largest node number.
    """
    # Flatten the list of tuples to a single list of integers
    nodes = [node for line in available_lines for node in line]
    
    # Return the largest number
    return max(nodes) if nodes else 0            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
