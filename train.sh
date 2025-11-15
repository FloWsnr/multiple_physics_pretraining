#!/bin/bash

### Task name
#SBATCH --account=sds_baek_energetic

### Job name
#SBATCH --job-name=train

### Output file
#SBATCH --output=results/00_slrm_logs/train_%j.out

### Number of nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --ntasks-per-node=70

### How much memory in total (MB)
#SBATCH --mem=300G

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zsa8rk@virginia.edu

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task (v100, a100, h200)
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb


### Partition
#SBATCH --partition=gpu

### create time series, i.e. 100 jobs one after another. Each runs for 24 hours
##SBATCH --array=1-10%1


#####################################################################################
############################# Setup #################################################
#####################################################################################

# activate conda environment
export CONDA_ROOT=$HOME/miniforge3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"
conda activate gphyt

ENV_FILE="/home/flwi01/coding/multiple_physics_pretraining/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi

######################################################################################
############################# Set paths ##############################################
######################################################################################

run_name="test01" # name of the folder where you placed the yaml config

python_exec="${BASE_DIR}/mpp/train_basic.py"
yaml_config="${BASE_DIR}/config/mpp_avit_b_custom.yaml"
config="basic_config"

nnodes=1
ngpus_per_node=1
export OMP_NUM_THREADS=1


#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
echo "--------------------------------"
echo "Starting training..."
echo "--------------------------------"

exec_args="--run_name $run_name --config $config --yaml_config $yaml_config --use_ddp"

# Capture Python output and errors in a variable and run the script
torchrun --standalone --nproc_per_node=$ngpus_per_node $python_exec $exec_args