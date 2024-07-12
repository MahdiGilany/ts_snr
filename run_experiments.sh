# # deeptime
# DATA="exchange_rate" # "exchange_rate" "etth2" "traffic" "electricity"
# LOOKBACK=96 # 288 336 672 720
# HORIZON=96 # 336 720
# DICT_NORM_COEF=0.0
# DICT_COV_COEF=0.0
# W_VAR_COEF=0.0
# W_COV_COEF=0.0
# TARGET=None
# # GROUP="deeptime_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# GROUP="deeptime_${DICT_NORM_COEF}dict-norm_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}_w-clp"
# # GROUP="deeptime_${DICT_COV_COEF}dict-cov_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# # GROUP="deeptime_${DICT_COV_COEF}dict-cov_${DICT_NORM_COEF}dict-norm_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}_v2"

# # GROUP="deeptime_${DICT_COV_COEF}dict-orth_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# # GROUP="deeptime_${DICT_COV_COEF}dict-orth_${DICT_NORM_COEF}dict-norm_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"

# # GROUP="deeptime_${W_VAR_COEF}var-${W_COV_COEF}cov-Wreg_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# # GROUP="deeptime_${W_VAR_COEF}var-${W_COV_COEF}cov-Wreg_${DICT_COV_COEF}dict-cov_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# for SEED in 0 1 2 3 4 # 5 6 7 8 9 
# do
#     python deeptime_experiment.py \
#         --name "${GROUP}_${SEED}" \
#         --group "${GROUP}" \
#         --cluster "slurm" \
#         --slurm_gres "gpu:a40:1" \
#         --seed $SEED \
#         --epochs 50 \
#         --patience 15 \
#         --layer_size 256 \
#         --dict_basis_norm_coeff $DICT_NORM_COEF \
#         --dict_basis_cov_coeff $DICT_COV_COEF \
#         --w_var_coeff $W_VAR_COEF \
#         --w_cov_coeff $W_COV_COEF \
#         --dataset_name $DATA \
#         --lookback $LOOKBACK \
#         --horizon $HORIZON \
#         --extend_val True \
#         --target_series_index $TARGET
# done 



# kernel deeptime
DATA="ettm2" # "exchange_rate" "etth2" "traffic" "electricity"
SUB_EXP="MultCheckScalesV2lrσ.001" 
LOOKBACK=960 # 288 336 672 720
HORIZON=192 # 336 720
TARGET=None
# GROUP="kernel-deeptime_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
GROUP="kernel-deeptime_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}_sub${SUB_EXP}"
for SEED in 0 1 2 # 3 4 # 5 6 7 8 9 
do
    python kernel_deeptime_experiment.py \
        --name "${GROUP}_${SEED}" \
        --group "${GROUP}" \
        --cluster "slurm" \
        --slurm_gres "gpu:rtx6000:1" \
        --seed $SEED \
        --epochs 50 \
        --patience 15 \
        --layer_size 256 \
        --inr_layers 5 \
        --patience 30 \
        --lr_σ 0.001 \
        --dataset_name $DATA \
        --lookback $LOOKBACK \
        --horizon $HORIZON \
        --extend_val True \
        --target_series_index $TARGET
done 


# # experiment
# EXP_CONFIG="kernel_deeptime"
# SUB_EXP="MultCheckV2" # V2=scalenorm
# NUM_SEEDS=5
# START_SEED=0
# LOOKBACK_MULT="None"
# DATA="ettm2" # "exchange_rate" "etth2" "traffic" "electricity"
# TARGET=None
# python experimenting.py \
#     --exp_config $EXP_CONFIG \
#     --sub_exp $SUB_EXP \
#     --num_seeds $NUM_SEEDS \
#     --start_seed $START_SEED \
#     --lookback_mult $LOOKBACK_MULT \
#     --cluster "slurm" \
#     --slurm_gres "gpu:a40:1" \
#     --epochs 50 \
#     --patience 25 \
#     --lr_σ 0.01 \
#     --layer_size 256 \
#     --dataset_name $DATA \
#     --extend_val True \
#     --target_series_index $TARGET
