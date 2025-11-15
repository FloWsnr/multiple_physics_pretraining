#!/usr/bin/bash

### Task name
#SBATCH --account=rwth1802
#SBATCH --job-name=train_mpp

### Output file
#SBATCH --output=results/slrm_logs/train_mpp_%j.out

### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --cpus-per-task=96
#SBATCH --exclusive

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=florian.wiesner@avt.rwth-aachen.de

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task
#SBATCH --gres=gpu:4



#####################################################################################
############################# Setup #################################################
#####################################################################################

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate gphyt

######################################################################################
############################# Set paths ##############################################
######################################################################################

run_name="mpp_02" # name of the folder where you placed the yaml config
base_dir="/hpcwork/rwth1802/coding/multiple_physics_pretraining"
data_dir="/hpcwork/rwth1802/coding/General-Physics-Transformer/data/datasets"
results_dir="${base_dir}/results"

python_exec="${base_dir}/mpp/train_basic.py"
yaml_config="${base_dir}/config/mpp_avit_b_custom.yaml"
config="basic_config"

nnodes=1
ngpus_per_node=4
export OMP_NUM_THREADS=4


#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
echo "--------------------------------"
echo "Starting training..."
echo "--------------------------------"

exec_args="--run_name $run_name --config $config --yaml_config $yaml_config --results_dir $results_dir --data_dir $data_dir --use_ddp"

# Capture Python output and errors in a variable and run the script
torchrun --standalone --nproc_per_node=$ngpus_per_node $python_exec $exec_args