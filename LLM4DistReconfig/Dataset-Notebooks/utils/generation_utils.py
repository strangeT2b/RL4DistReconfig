import torch
from transformers import GenerationConfig
from time import perf_counter
import re
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM

def generate_response(user_input, model, tokenizer, max_new_tokens, penalty_alpha=0.6, do_sample=True, top_k=5, temperature=0.5, repetition_penalty=1.2, skip_special_tokens=True):

    prompt = formatted_prompt(user_input)
    
    inputs = tokenizer([prompt], return_tensors="pt")
    
    generation_config = GenerationConfig(
      penalty_alpha=penalty_alpha,
      do_sample = do_sample,
      top_k=top_k,
      temperature=temperature,
      repetition_penalty=repetition_penalty,
      max_new_tokens=max_new_tokens,
      pad_token_id=tokenizer.eos_token_id, 
      eos_token_id=tokenizer.eos_token_id  # Ensure EOS token is set - This is a new parameter so maybe it breaks the code.
    )
    start_time = perf_counter()
    
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    
    outputs = model.generate(**inputs, generation_config=generation_config)
    response = tokenizer.decode(outputs[0], skip_special_tokens=skip_special_tokens)
    output_time = perf_counter() - start_time
    return response, output_time
    
    
def formatted_prompt(question)-> str:
    return f"<|user|>\n{question}</s>\n<|assistant|>"
    


def extract_output_data(response):
    """
    Extracts the output data related to the open lines, node voltages, and system loss from the response string.
    
    Parameters:
    - response: The full response string from which to extract the output data.
    
    Returns:
    A dictionary containing the extracted output data.
    """
    # Regular expression to match the output section
    output_pattern = re.compile(
        r"Open Lines=\[(.*?)\],\s*(?:Node Voltages|Network Variables: NodeVoltages)=\[(.*?)\],\s*System Loss=(\d+\.\d+)"
    )

    # Search for the output pattern in the response
    match = output_pattern.search(response)

    if match:
        # Extract groups from the match
        open_lines = match.group(1)
        node_voltages = match.group(2)
        system_loss = match.group(3)
        
        # Debugging prints
        print(f"Extracted open lines: {open_lines}")
        print(f"Extracted node voltages: {node_voltages}")
        print(f"Extracted system loss: {system_loss}")

        # Fix any missing parentheses for the first and last open lines
        if not open_lines.startswith('('):
            open_lines = '(' + open_lines
        if not open_lines.endswith(')'):
            open_lines = open_lines + ')'

        # Properly format the open_lines string to remove extra characters
        try:
            open_lines_list = [tuple(map(int, line.split(','))) for line in re.findall(r'\((\d+,\s*\d+)\)', open_lines)]
        except ValueError as e:
            print(f"Error processing open lines: {e}")
            open_lines_list = []

        try:
            node_voltages_list = [float(voltage.strip()) for voltage in node_voltages.split(',')]
        except ValueError as e:
            print(f"Error processing node voltages: {e}")
            node_voltages_list = []

        # Debugging prints
        print(f"Formatted open lines list: {open_lines_list}")
        print(f"Formatted node voltages list: {node_voltages_list}")

        # Return the extracted data in a structured format
        return {
            'Open Lines': open_lines_list,
            'Node Voltages': node_voltages_list,
            'System Loss': float(system_loss)
        }
    else:
        # Debugging print
        print(f"No match found in response: {response}")
        return "No output data found in the response."


        
  
        
        
def generate_and_save_responses(dataset, response_function, num_samples=10, filename="responses.txt"):
    """
    Generates responses for randomly selected prompts from the dataset, includes the correct output,
    and saves them to a file with clear separation between each sample.
    
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
    indices = torch.randint(0, len(dataset), (num_samples,))
    
    # Open the file to write the responses
    with open(filename, "w") as file:
        for index in indices:
            # Extract the prompt and correct output using the random index
            prompt = dataset['prompt'][index]
            correct_output = dataset['output'][index]
            
            # Generate the response
            response, output_time = response_function(user_input=prompt)

            reformatted_response = extract_output_data(response)
            
            # Write the prompt, response, and correct output to the file with clear separation
            file.write("---- Sample Start ----\n")
            file.write(f"Prompt:\n {prompt}\n")
            file.write(f"Response:\n {reformatted_response}\n")
            file.write(f"Correct Output:\n {correct_output}\n")
            file.write(f"Inference Time: {output_time}\n")
            file.write("---- Sample End ----\n\n")        
        
        
def peft_merge_unload(model_id,model_path, torch_dtype=torch.float16, load_in_8bit=False, device_map="auto", trust_remote_code=True, from_transformers=True):
    
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_id, 
                                            torch_dtype=torch_dtype, 
                                            load_in_8bit=load_in_8bit,
                                             device_map=device_map,
                                             trust_remote_code=trust_remote_code)
                                             
    peft_model = PeftModel.from_pretrained(model = model, model_id=model_path, from_transformers=from_transformers, device_map=device_map)
    
    model = peft_model.merge_and_unload()
    return model




        
        
        
        
        
        
        
        
        