#!/bin/bash

for noise_std in 0 10 50 100 200
do
 for seed in {0..4}
 do
 echo "seed ${seed} and noise std ${noise_std} main experiment nbeats"
 python main.py experiment=nbeats seed=${seed}\
 data.noise_std=${noise_std}\
 name="nbeats_airpassenger_noisestd${noise_std}_seed${seed}"\
 logger.wandb.group=nbeats_noise_std${noise_std}
 done
done

# for noise_std in 0 10 50 100 200
# do
#  for seed in {0..4}
#  do
#  echo "seed ${seed} and noise std ${noise_std} main experiment deeptime"
#  python main.py experiment=deeptime seed=${seed}\
#  data.noise_std=${noise_std}\
#  name="deeptime_airpassenger_noisestd${noise_std}_seed${seed}"\
#  logger.wandb.group=deeptime_noise_std${noise_std}
#  done
# done