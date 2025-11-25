#!/usr/bin/bash

### Task name
#SBATCH --account=rwth1802
#SBATCH --job-name=eval_mpp_novel

### Output file
#SBATCH --output=results/slrm_logs/eval_mpp_novel_%j.out

### Start a parallel job for a distributed-memory system on several nodes
#SBATCH --nodes=1

### How many CPU cores to use
#SBATCH --cpus-per-task=24
##SBATCH --exclusive

### Mail notification configuration
#SBATCH --mail-type=ALL
#SBATCH --mail-user=florian.wiesner@avt.rwth-aachen.de

### Maximum runtime per task
#SBATCH --time=24:00:00

### set number of GPUs per task
#SBATCH --gres=gpu:1


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
# debug mode
# debug=true
# Set up paths
base_dir="/hpcwork/rwth1802/coding/multiple_physics_pretraining"
python_bin="/home/fw641779/miniforge3/envs/gphyt/bin/python"
python_exec="${base_dir}/mpp/model_eval.py"
eval_dir="${base_dir}/results"

data_dir="/hpcwork/rwth1802/coding/General-Physics-Transformer/data/datasets"
sim_name="mpp_04"
# name of the checkpoint to use for evaluation.
checkpoint_name="ckpt.tar_epoch20"
# forcasts
forecast="1 4 8 12 16 20 24"
# subdir name
sub_dir="eval/all_horizons_novel"
debug=false



nnodes=1
ngpus_per_node=1
export OMP_NUM_THREADS=1


# sim directory
sim_dir="${eval_dir}/${sim_name}"



#######################################################################################
############################# Setup sim dir and config file ###########################
#######################################################################################

# create the sim_dir if it doesn't exist
mkdir -p $sim_dir
# Try to find config file in sim_dir
config_file="${base_dir}/config/mpp_avit_b_eval_novel.yaml"
if [ ! -f "$config_file" ]; then
    echo "No config_eval.yaml file found in $config_file, aborting..."
    exit 1
fi


#####################################################################################
############################# Evaluation ############################################
#####################################################################################
echo "--------------------------------"
echo "Starting evaluation..."
echo "config_file: $config_file"
echo "sim_dir: $sim_dir"
echo "using checkpoint: $checkpoint_name"
echo "--------------------------------"

exec_args="--config_file $config_file \
    --sim_dir $sim_dir \
    --data_dir $data_dir \
    --forecast_horizons $forecast \
    --checkpoint_name $checkpoint_name \
    --subdir_name $sub_dir"

if [ "$debug" = true ]; then
    echo "Running in debug mode."
    exec_args="$exec_args --debug"
fi

# Capture Python output and errors in a variable and run the script
$python_bin $python_exec $exec_args

# move the output file to the sim_dir
mv ${sim_dir}/slrm_logs/eval_${sim_name}_${SLURM_JOB_ID}.out $sim_dir