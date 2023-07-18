#!/bin/bash


# # deeptime
# sbatch scripts/skynet_noise.sh version=benchmarkingV2 model_name=deeptime dataset_name=etth2 target_series_index=-1 multiple=7 output_chunk_length=96

# sbatch scripts/skynet_noise.sh version=benchmarkingV2 model_name=deeptime dataset_name=exchange_rate target_series_index=-2 multiple=3 output_chunk_length=96

# sbatch scripts/skynet_noise.sh version=benchmarkingV2 model_name=deeptime dataset_name=traffic target_series_index=-1 multiple=9 output_chunk_length=96


# # nbeats
# sbatch scripts/skynet_noise.sh version=benchmarkingV3 model_name=nbeats dataset_name=etth2 target_series_index=-1 multiple=7 output_chunk_length=96

# sbatch scripts/skynet_noise.sh version=benchmarkingV3 model_name=nbeats dataset_name=ettm2 target_series_index=-1 multiple=7 output_chunk_length=96

# sbatch scripts/skynet_noise.sh version=benchmarkingV2 model_name=nbeats dataset_name=exchange_rate target_series_index=-2 multiple=3 output_chunk_length=96
# sbatch scripts/skynet_noise_temp.sh version=benchmarkingV3genericFalse model_name=nbeats dataset_name=exchange_rate target_series_index=-2 multiple=1 output_chunk_length=96

# # sbatch scripts/skynet_noise.sh version=benchmarkingV2 model_name=nbeats dataset_name=traffic target_series_index=-1 multiple=9 output_chunk_length=96


# # # omp_deeptime
# sbatch scripts/skynet_noise_OMP.sh version=benchmarkingV2 model_name=omp_deeptime dataset_name=etth2 target_series_index=-1 multiple=7 output_chunk_length=96 n_nonzero_coefs=15

# sbatch scripts/skynet_noise_OMP.sh version=benchmarkingV2 model_name=omp_deeptime dataset_name=exchange_rate target_series_index=-2 multiple=3 output_chunk_length=96 n_nonzero_coefs=15

# sbatch scripts/skynet_noise_OMP.sh version=benchmarkingV2 model_name=omp_deeptime dataset_name=traffic target_series_index=-1 multiple=9 output_chunk_length=96 n_nonzero_coefs=15


# # # naive_martingle
# # sbatch scripts/skynet_noise_naive.sh version=benchmarkingV2 model_name=naive_martingle dataset_name=etth2 target_series_index=-1 multiple=1 output_chunk_length=96

# # sbatch scripts/skynet_noise_naive.sh version=benchmarkingV2 model_name=naive_martingle dataset_name=exchange_rate target_series_index=-2 multiple=1 output_chunk_length=96

# # sbatch scripts/skynet_noise_naive.sh version=benchmarkingV2 model_name=naive_martingle dataset_name=traffic target_series_index=-1 multiple=1 output_chunk_length=96


# # # naive_moving_average
# # sbatch scripts/skynet_noise_naive.sh version=benchmarkingV2 model_name=naive_movingaverage dataset_name=etth2 target_series_index=-1 multiple=7 output_chunk_length=96

# # sbatch scripts/skynet_noise_naive.sh version=benchmarkingV2 model_name=naive_movingaverage dataset_name=exchange_rate target_series_index=-2 multiple=3 output_chunk_length=96

# # sbatch scripts/skynet_noise_naive.sh version=benchmarkingV2 model_name=naive_movingaverage dataset_name=traffic target_series_index=-1 multiple=9 output_chunk_length=96


# # # naive_seasonal
# # sbatch scripts/skynet_noise_naive.sh version=benchmarkingV2 model_name=naive_seasonal dataset_name=etth2 target_series_index=-1 multiple=1 output_chunk_length=96

# # sbatch scripts/skynet_noise_naive.sh version=benchmarkingV2 model_name=naive_seasonal dataset_name=exchange_rate target_series_index=-2 multiple=1 output_chunk_length=96

# # sbatch scripts/skynet_noise_naive.sh version=benchmarkingV2 model_name=naive_seasonal dataset_name=traffic target_series_index=-1 multiple=1 output_chunk_length=96


# for n_nonzero_coefs in 2 5 10 15 45 75 105 135 #165 195 #225 255
# do
#     # omp_deeptime
#     sbatch scripts/skynet_noise_OMP.sh version=benchmarkingV4lambda_5 model_name=omp_deeptime dataset_name=etth2 target_series_index=-1 multiple=7 output_chunk_length=96 n_nonzero_coefs=$n_nonzero_coefs
#     # sbatch scripts/skynet_noise_OMP.sh version=benchmarkingV4lambda_5 model_name=omp_deeptime dataset_name=exchange_rate target_series_index=-2 multiple=3 output_chunk_length=96 n_nonzero_coefs=$n_nonzero_coefs
#     # sbatch scripts/skynet_noise_OMP.sh version=benchmarkingV4lambda_5 model_name=omp_deeptime dataset_name=traffic target_series_index=-1 multiple=9 output_chunk_length=96 n_nonzero_coefs=$n_nonzero_coefs
# done

# for multiple in 1 3 5 7 9
# do 
#     sbatch scripts/skynet_noise_without_loop.sh version=ablation_multiple model_name=deeptime dataset_name=etth2 target_series_index=-1 multiple=$multiple output_chunk_length=96
# done