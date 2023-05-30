#!/bin/bash
version=+testset
for seed in {0..4}
do
 for noise_std in 0 10 50 100 200
 do
 echo "seed ${seed} and noise std ${noise_std} main experiment deeptime"
 python main.py experiment=deeptime seed=${seed}\
 data.noise_std=${noise_std}\
 name="deeptime_airpassenger_noisestd${noise_std}_seed${seed}_v${version}"\
 logger.wandb.group=deeptime_noise_std${noise_std}_v${version}
 done
done

for seed in {0..4}
do
 for noise_std in 0 10 50 100 200
 do
 echo "seed ${seed} and noise std ${noise_std} main experiment nbeats"
 python main.py experiment=nbeats seed=${seed}\
 data.noise_std=${noise_std}\
 name="nbeats_airpassenger_noisestd${noise_std}_seed${seed}_v${version}"\
 logger.wandb.group=nbeats_noise_std${noise_std}_v${version}
 done
done
