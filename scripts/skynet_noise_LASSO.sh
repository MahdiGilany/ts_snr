#!/bin/bash

#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --time=3-01:00:00
#SBATCH --exclude=compute1080ti01
#SBATCH -c 6 
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
# sleep 3

# HYDRA_FULL_ERROR=1
# export HYDRA_FULL_ERROR


echo "STARTING"

# defaults
version=Slurm_LASSO_adpt5
experiment="exp_default"
# model_name="maml_deeptime"
model_name="LASSO_deeptime"
seed=0
batch_size=256
epochs=100
lr=0.001
adapt_steps=5
adapt_lr=0.01
L1=True
dataset_name="exchange_rate"
# only for crypto
crypto_name="Bitcoin"
prct_rows_to_load=0.1
#
noise_type="gaussian"
noise_std=0
# target_series_index=-1
outdim=8
new_dir=True
verbose=False
multiple=3
output_chunk_length=96
input_chunk_length=-1

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

for seed in {0..0}
do

rand_seed=$((seed + SLURM_JOB_ID))

for noise_std in 0 #0.5 1.0 # 0 0.3 0.6 0.9 1.2 1.5 1.8 #2.0 2.5 3.0 3.5
do

# set name, group, and input chunk length
if [ $input_chunk_length == -1 ]
then
    input_chunk_length=$((output_chunk_length * multiple))
fi
group="${model_name}_${dataset_name}_in${input_chunk_length}_out${output_chunk_length}_noise_${noise_type}_std${noise_std}_L1${L1}_v${version}"
name="${group}_seed${rand_seed}"
            # data.target_series_index=$target_series_index\
echo "seed ${rand_seed} and noise std ${noise_std} model_name ${model_name}"
python main.py name=$name\
            experiment=$experiment\
            seed=$rand_seed\
            batch_size=$batch_size\
            epochs=$epochs\
            new_dir=$new_dir\
            model.model_name=$model_name\
            model.input_chunk_length=$input_chunk_length\
            model.output_chunk_length=$output_chunk_length\
            model.optimizer_kwargs.lr=$lr\
            +model.adaptation_steps=$adapt_steps\
            +model.adaptation_lr=$adapt_lr\
            +model.L1=$L1\
            +model.out_dim=$outdim\
            data.dataset_name=$dataset_name\
            +data.crypto_name=$crypto_name\
            +data.prct_rows_to_load=$prct_rows_to_load\
            data.noise_type=$noise_type\
            data.noise_std=$noise_std\
            logger.wandb.group=$group\
            verbose=$verbose\
            id=$SLURM_JOB_ID
done
done

echo "DONE"
