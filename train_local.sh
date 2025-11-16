#!/bin/bash

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

run_name="mpp_test02" # name of the folder where you placed the yaml config
base_dir="/home/flwi01/coding/multiple_physics_pretraining"
data_dir="/home/flwi01/coding/well_datasets"
results_dir="${base_dir}/results"

python_exec="${base_dir}/mpp/train_basic.py"
yaml_config="${base_dir}/config/mpp_avit_b_custom.yaml"
config="basic_config"

nnodes=1
ngpus_per_node=1
export OMP_NUM_THREADS=4


#####################################################################################
############################# Training GPM ##########################################
#####################################################################################
echo "--------------------------------"
echo "Starting training..."
echo "--------------------------------"

exec_args="--run_name $run_name --config $config --yaml_config $yaml_config --results_dir $results_dir --data_dir $data_dir"

# Capture Python output and errors in a variable and run the script
$HOME/miniforge3/envs/gphyt/bin/python $python_exec $exec_args