#!/bin/bash

output_chunk_length=96
noise_type="laplace"
version=benchmarkingV2
omp_version=benchmarkingV4lambda_5

for model_name in deeptime nbeats omp_deeptime #naive_martingle 
do
    for dataset_name in etth2 exchange_rate traffic #ettm2
    do
        # multiple
        if [ $dataset_name == "etth2" ]
        then
            multiple=7
            target_index=-1
        elif [ $dataset_name == "exchange_rate" ]
        then
            multiple=3
            target_index=-2
        else
            multiple=9
            target_index=-1
        fi

        if [ $model_name == "omp_deeptime" ]
        then
            for n_nonzero_coefs in 2 5 10 15 45 75 105 135 #165 195 #225 255
            do
                sbatch scripts/skynet_noise_OMP.sh version=$omp_version model_name=$model_name dataset_name=$dataset_name\
                target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type\
                n_nonzero_coefs=$n_nonzero_coefs        
            done
        else
            if [ $model_name == "nbeats" ] && [ $dataset_name == "exchange_rate" ]
            then
                sbatch scripts/skynet_noise_temp.sh version=${version}genericFalse model_name=$model_name dataset_name=$dataset_name\
                 target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type
            else
                sbatch scripts/skynet_noise.sh version=$version model_name=$model_name dataset_name=$dataset_name\
                target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type
            fi

        fi

    done
done

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


# # naive_martingle
# sbatch scripts/skynet_noise_naive.sh version=benchmarkingV3 model_name=naive_martingle dataset_name=etth2 target_series_index=-1 multiple=1 output_chunk_length=96

# sbatch scripts/skynet_noise_naive.sh version=benchmarkingV3 model_name=naive_martingle dataset_name=exchange_rate target_series_index=-2 multiple=1 output_chunk_length=96

# sbatch scripts/skynet_noise_naive.sh version=benchmarkingV3 model_name=naive_martingle dataset_name=traffic target_series_index=-1 multiple=1 output_chunk_length=96


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


# omp thresholding
# for tolerance in 0.01 0.05 0.1 0.15 0.2 0.25 0.3 
# do
    # sbatch scripts/skynet_noise_without_loop_OMP.sh version=thresholding_check model_name=omp_deeptime dataset_name=etth2\
    #  target_series_index=-1 multiple=7 output_chunk_length=96 tolerance=$tolerance n_nonzero_coefs=135 epochs=40

    # sbatch scripts/skynet_noise_without_loop_OMP.sh version=thresholding_check model_name=omp_deeptime dataset_name=exchange_rate\
    #  target_series_index=-2 multiple=3 output_chunk_length=96 tolerance=$tolerance n_nonzero_coefs=135 epochs=40
    
    # sbatch scripts/skynet_noise_without_loop_OMP.sh version=thresholding_check model_name=omp_deeptime dataset_name=traffic\
    #  target_series_index=-1 multiple=9 output_chunk_length=96 tolerance=$tolerance n_nonzero_coefs=135 epochs=40
# done