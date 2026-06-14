import torch
import subprocess as sp
import os
import nvidia_smi


def get_memory_allocated_cached():
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MiB")
    print(f"Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 2} MiB")
    
def get_cuda_info():
    if torch.cuda.is_available():
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__CUDA Device Name:', torch.cuda.get_device_name(0))
        print('__CUDA Device Total Memory [GB]:', torch.cuda.get_device_properties(0).total_memory / 1e9)
    else:
        print('CUDA is not available')
        
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values
    
def get_nvidia_smi_info(index):
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(index)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print("Total memory:", info.total / 1e9, "GB")
    print("Free memory:", info.free / 1e9, "GB")
    print("Used memory:", info.used / 1e9, "GB")

    nvidia_smi.nvmlShutdown()