#!/bin/bash

for noise_std in 0 10 50 100 200
do
 for seed in {0..4}
 do
 echo "seed ${seed} and noise std ${noise_std} main experiment exp_default"
 python main.py experiment=exp_default seed=${seed}\
 data.noise_std=${noise_std}\
 name="nbeats_airpassenger_noisestd${noise_std}_seed${seed}"\
 logger.wandb.group=noise_std${noise_std}
 done
done