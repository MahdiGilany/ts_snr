#!/bin/bash

#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --time=1-01:00:00
#SBATCH --exclude=compute1080ti06,compute1080ti08
#SBATCH -c 16 
#SBATCH -o /home/abbasgln/code/ts_snr/slurm_logs/%J.out
#SBATCH -e /home/abbasgln/code/ts_snr/slurm_logs/%J.err 


source /home/abbasgln/anaconda3/bin/activate borealis

hostname
whoami
echo "Job_ID="$SLURM_JOB_ID
echo "CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES
# nvidia-smi
# rmmod nvidia_uvm
# modprobe nvidia_uvm
sleep 3

# HYDRA_FULL_ERROR=1
# export HYDRA_FULL_ERROR


echo "STARTING"

# defaults
version=Slurm_OT
experiment="exp_default"
driver=localforecasting_driver
model_name="naive_seasonal"
seed=0
batch_size=256
epochs=100
lr=0.001
dataset_name="etth2"
noise_std=0
target_series_index=-1
new_dir=True
verbose=False
multiple=1
output_chunk_length=96

# parse arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# run experiment
# python scripts/test.py

for noise_std in 0 0.1 0.3 0.5 0.7 0.9 #1.1 1.3 1.5 2.0
do

for seed in {0..0}
do

# set name, group, and input chunk length
input_chunk_length=$((output_chunk_length * multiple))
group="${model_name}_${dataset_name}_in${input_chunk_length}_out${output_chunk_length}_noise_std${noise_std}_v${version}"
name="${group}_seed${seed}"

echo "seed ${seed} and noise std ${noise_std} model_name ${model_name}"
python main.py name=$name\
            experiment=$experiment\
            seed=$seed\
            batch_size=$batch_size\
            epochs=$epochs\
            new_dir=$new_dir\
            driver=$driver\
            model.model_name=$model_name\
            model.input_chunk_length=$input_chunk_length\
            model.output_chunk_length=$output_chunk_length\
            model.optimizer_kwargs.lr=$lr\
            data.dataset_name=$dataset_name\
            data.noise_std=$noise_std\
            data.target_series_index=$target_series_index\
            logger.wandb.group=$group\
            verbose=$verbose\
            id=$SLURM_JOB_ID
done
done

echo "DONE"
