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
python main.py experiment=deeptime new_dir=True id=$SLURM_JOB_ID epochs=100
echo "DONE"
