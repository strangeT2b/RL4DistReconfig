#!/bin/bash

#SBATCH --output="/scratch/pc2442/LLM-Reconfiguration/AutoTrain/llama2/llama2-finetuning-untrained-GET-METRICS-84N.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=200GB
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=llama2-finetuning-untrained-GET-METRICS-84N



module purge
module load intel/19.1.2
module load anaconda3/2020.07
module load python/intel/3.8.6

cd /scratch/pc2442/LLM-Reconfiguration/AutoTrain/Finetuned-Models/

# Print the SLURM job configurations
echo "SLURM Job Configuration:"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Memory per Node: $SLURM_MEM_PER_NODE"
echo "Time Limit: $SLURM_TIMELIMIT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Output File: $SLURM_JOB_OUT"

echo "Hardware Configuration"
echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep Mem: | awk '{print $4}')"
echo "GPU: $(nvidia-smi -q | grep 'Product Name')"

singularity exec --nv --overlay /scratch/pc2442/pytorch-example/my_pytorch.ext3:ro /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash -c \
            "source /ext3/env.sh;  python /scratch/pc2442/LLM-Reconfiguration/Model-Notebooks/llama3-notebooks/generate-metrics.py \
            --data_path /scratch/pc2442/LLM-Reconfiguration/Dataset-Notebooks/train_files/train_84_nodes.csv \
            --model_id meta-llama/Llama-2-7b-hf \
            --model_for_generation_path meta-llama/Llama-2-7b-hf \
            --max_new_tokens 1200 \
            --filename_txt /scratch/pc2442/LLM-Reconfiguration/Dataset-Notebooks/evaluations/metrics/llama2-grid-reconfiguration-untrained-testset-metrics-n500-84N.txt\
            --filename_csv /scratch/pc2442/LLM-Reconfiguration/Dataset-Notebooks/evaluations/responses/llama2-grid-reconfiguration-untrained-testset-metrics-n500-84N.csv\
            --num_samples 500 "