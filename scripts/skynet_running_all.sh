#!/bin/bash

output_chunk_length=96
noise_type="laplace"
version=sparsity_benchmarking
omp_version=sparsity_benchmarking

# defaults
crypto_name="Bitcoin"
# crypto_name="Ethereum"
# crypto_name="Litecoin"
prct_rows_to_load=0.1

# with loop over noise and seeds
for model_name in  deeptime nbeats omp_deeptime naive_martingle mixture_experts_deeptime  # l1_deeptime vd_deeptime # inrplay_deeptime(NeRF) 
do
    for dataset_name in etth2 exchange_rate traffic #crypto 
    do

        # multiple
        if [ $dataset_name == "etth2" ]
        then
            multiple=3
            target_index=-1
            n_nonzero_coefs=105
        elif [ $dataset_name == "exchange_rate" ]
        then
            multiple=3
            target_index=-2
            n_nonzero_coefs=45
        elif [ $dataset_name == "traffic" ]
        then
            multiple=7
            target_index=-1
            n_nonzero_coefs=105
        elif [ $dataset_name == "crypto" ]
        then
            multiple=7
            target_index=-1
            n_nonzero_coefs=45
            output_chunk_length=15 # make sure crypto is the last dataset since changes state of output_chunk_length
            version=${version}_${crypto_name}_prct${prct_rows_to_load}
            omp_version=${omp_version}_${crypto_name}_prct${prct_rows_to_load}
        else    
            multiple=7
            target_index=-1
            n_nonzero_coefs=45
        fi


        if [ $model_name == "omp_deeptime" ]
        then
            # for n_nonzero_coefs in 2 5 10 15 45 75 105 135 #165 195 #225 255
            # do
                sbatch scripts/skynet_noise_OMP.sh version=$omp_version model_name=$model_name dataset_name=$dataset_name\
                target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type\
                n_nonzero_coefs=$n_nonzero_coefs crypto_name=$crypto_name prct_rows_to_load=$prct_rows_to_load       
            # done
        elif [ $model_name == "naive_martingle" ]
        then
            sbatch scripts/skynet_noise_naive.sh version=$version model_name=$model_name dataset_name=$dataset_name\
            target_series_index=$target_index multiple=1 output_chunk_length=$output_chunk_length noise_type=$noise_type\
            crypto_name=$crypto_name prct_rows_to_load=$prct_rows_to_load 
        elif [ $model_name == "mixture_experts_deeptime" ]
        then
            sbatch scripts/skynet_noise_MOE.sh version=$version model_name=$model_name dataset_name=$dataset_name\
            target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type\
            K_value=$n_nonzero_coefs crypto_name=$crypto_name prct_rows_to_load=$prct_rows_to_load 
        else
            if [ $model_name == "nbeats" ] && [ $dataset_name == "exchange_rate" ]
            then
                sbatch scripts/skynet_noise_temp.sh version=${version}genericFalse model_name=$model_name dataset_name=$dataset_name\
                 target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type\
                 crypto_name=$crypto_name prct_rows_to_load=$prct_rows_to_load 
            else
                sbatch scripts/skynet_noise.sh version=$version model_name=$model_name dataset_name=$dataset_name\
                target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type\
                crypto_name=$crypto_name prct_rows_to_load=$prct_rows_to_load 
            fi

        fi

    done
done


# # without loop over noise and seeds
# for model_name in deeptime nbeats omp_deeptime #naive_martingle 
# do
#     for dataset_name in etth2 exchange_rate traffic #ettm2
#     do
#     for seed in {0..2}
#     do
#     for noise_std in 0.9
#     do
#         # multiple
#         if [ $dataset_name == "etth2" ]
#         then
#             multiple=7
#             target_index=-1
#         elif [ $dataset_name == "exchange_rate" ]
#         then
#             multiple=3
#             target_index=-2
#         else
#             multiple=9
#             target_index=-1
#         fi

#         # for input_chunk_length in 96 288 480 672 #864 #1056
#         # do
#         # for multiple in $multiple #1 3 5 7 9
#         # do
#             if [ $model_name == "omp_deeptime" ] 
#             then
#                 for n_nonzero_coefs in 45 # 2 5 10 15 45 75 105 135 #165 195 #225 255
#                 do
#                     sbatch scripts/skynet_noise_without_loop_OMP.sh version=$omp_version model_name=$model_name dataset_name=$dataset_name\
#                     target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type\
#                     noise_std=$noise_std n_nonzero_coefs=$n_nonzero_coefs seed=$seed input_chunk_length=$input_chunk_length
#                 done
#             else
#                 if [ $model_name == "nbeats" ] && [ $dataset_name == "exchange_rate" ]
#                 then
#                     placeholder=0
#                     # sbatch scripts/skynet_noise_without_loop_temp.sh version=${version}genericFalse model_name=$model_name dataset_name=$dataset_name\
#                     #  target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type\
#                     #  noise_std=$noise_std seed=$seed # input_chunk_length=$input_chunk_length
#                 else
#                     sbatch scripts/skynet_noise_without_loop.sh version=$version model_name=$model_name dataset_name=$dataset_name\
#                     target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type\
#                     noise_std=$noise_std seed=$seed # input_chunk_length=$input_chunk_length
#                 fi

#             fi
#         # done

#     done
#     done
#     done
# done


# # test MAE
# omp_version=test_MAE
# n_nonzero_coefs=45
# model_name=omp_deeptime
# dataset_name=etth2
# target_index=-1
# multiple=7
# output_chunk_length=96
# noise_type=laplace
# noise_std=1.2
# loss_name=mae

# sbatch scripts/skynet_noise_OMP.sh version=$omp_version model_name=$model_name dataset_name=$dataset_name\
#  target_series_index=$target_index multiple=$multiple output_chunk_length=$output_chunk_length noise_type=$noise_type\
#  noise_std=$noise_std n_nonzero_coefs=$n_nonzero_coefs loss_fn=$loss_name 




######################################################################################
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