#!/bin/bash

#SBATCH --mem=25G
#SBATCH --gres=gpu:1
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
version=Slurm
experiment="nbeats"
seed=0
batch_size=256
epochs=300
noise_std=0
dataset_name="etth2"
new_dir=False
verbose=False
multiple=7
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

# for noise_std in 0 .1 .3 .5 .7 .9 2.0
# do

for seed in {0..4}
do

# set name, group, and input chunk length
input_chunk_length=$((output_chunk_length * multiple))
group="${experiment}_${dataset_name}_in${input_chunk_length}_out${output_chunk_length}_noise_std${noise_std}_v${version}"
name="${group}_seed${seed}"

echo "seed ${seed} and noise std ${noise_std} experiment ${experiment}"
python main.py name=$name\
            experiment=$experiment\
            seed=$seed\
            batch_size=$batch_size\
            epochs=$epochs\
            new_dir=$new_dir\
            model.input_chunk_length=$input_chunk_length\
            model.output_chunk_length=$output_chunk_length\
            data.dataset_name=$dataset_name\
            data.noise_std=$noise_std\
            logger.wandb.group=$group\
            verbose=$verbose\
            id=$SLURM_JOB_ID
done

echo "DONE"
