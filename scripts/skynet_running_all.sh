#!/bin/bash


# deeptime
sbatch scripts/skynet_noise.sh version=benchmarking model_name=deeptime dataset_name=etth2 target_series_index=-1 multiple=7 output_chunk_length=96

sbatch scripts/skynet_noise.sh version=benchmarking model_name=deeptime dataset_name=exchange_rate target_series_index=-1 multiple=3 output_chunk_length=96

sbatch scripts/skynet_noise.sh version=benchmarking model_name=deeptime dataset_name=traffic target_series_index=-1 multiple=9 output_chunk_length=96


# nbeats
sbatch scripts/skynet_noise.sh version=benchmarking model_name=nbeats dataset_name=etth2 target_series_index=-1 multiple=7 output_chunk_length=96

sbatch scripts/skynet_noise.sh version=benchmarking model_name=nbeats dataset_name=exchange_rate target_series_index=-1 multiple=3 output_chunk_length=96

# sbatch scripts/skynet_noise.sh version=benchmarking model_name=nbeats dataset_name=traffic target_series_index=-1 multiple=9 output_chunk_length=96


# # omp_deeptime
# sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=omp_deeptime dataset_name=etth2 target_series_index=-1 multiple=7 output_chunk_length=96 n_nonzero_coefs=15

# sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=omp_deeptime dataset_name=exchange_rate target_series_index=-1 multiple=3 output_chunk_length=96 n_nonzero_coefs=15

# sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=omp_deeptime dataset_name=traffic target_series_index=-1 multiple=9 output_chunk_length=96 n_nonzero_coefs=15


# naive_martingle
sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=naive_martingle dataset_name=etth2 target_series_index=-1 multiple=1 output_chunk_length=96

sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=naive_martingle dataset_name=exchange_rate target_series_index=-1 multiple=1 output_chunk_length=96

sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=naive_martingle dataset_name=traffic target_series_index=-1 multiple=1 output_chunk_length=96


# naive_moving_average
sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=naive_movingaverage dataset_name=etth2 target_series_index=-1 multiple=7 output_chunk_length=96

sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=naive_movingaverage dataset_name=exchange_rate target_series_index=-1 multiple=3 output_chunk_length=96

sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=naive_movingaverage dataset_name=traffic target_series_index=-1 multiple=9 output_chunk_length=96


# # naive_seasonal
# sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=naive_seasonal dataset_name=etth2 target_series_index=-1 multiple=1 output_chunk_length=96

# sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=naive_seasonal dataset_name=exchange_rate target_series_index=-1 multiple=1 output_chunk_length=96

# sbatch scripts/skynet_noise_naive.sh version=benchmarking model_name=naive_seasonal dataset_name=traffic target_series_index=-1 multiple=1 output_chunk_length=96


