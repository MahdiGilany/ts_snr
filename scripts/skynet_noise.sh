#!/bin/bash

#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH -c 16 
#SBATCH -o /home/abbasgln/code/ts_snr/slurm_logs/%J.out
#SBATCH -e /home/abbasgln/code/ts_snr/slurm_logs/%J.err 


source /home/abbasgln/anaconda3/bin/activate borealis

hostname
whoami
echo "Job_ID="$SLURM_JOB_ID

echo $CUDA_VISIBLE_DEVICES
sleep 10


echo "STARTING"

# defaults
version=_slurm
experiment="nbeats"
seed=0
batch_size=256
epochs=10
noise_std=0
dataset_name="etth2"
new_dir=False
multiple=7
output_chunk_length=96
input_chunk_length=$((output_chunk_length * multiple))

group="${experiment}_${dataset_name}_seed${seed}_v${version}"
name="noisestd${noise_std}_${group}"

# parse arguments
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# run experiment
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
            verbose=False\
            id=$SLURM_JOB_ID\


echo "DONE"
