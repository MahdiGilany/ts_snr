# deeptime
DATA="exchange_rate" # "exchange_rate" "etth2" "traffic" "electricity"
LOOKBACK=2352 # 288 336 672 720
HORIZON=336 # 336 720
DICT_NORM_COEF=0.5
DICT_COV_COEF=1.0
TARGET=None
# GROUP="deeptime_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# GROUP="deeptime_${DICT_NORM_COEF}dict-norm_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# GROUP="deeptime_${DICT_COV_COEF}dict-cov_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
GROUP="deeptime_${DICT_COV_COEF}dict-cov_${DICT_NORM_COEF}dict-norm_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}_v2"

# GROUP="deeptime_${DICT_COV_COEF}dict-orth_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# GROUP="deeptime_${DICT_COV_COEF}dict-orth_${DICT_NORM_COEF}dict-norm_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
for SEED in 0 1 2 #3 4 # 5 6 7 8 9 
do
    python deeptime_experiment.py \
        --name "${GROUP}_${SEED}" \
        --group "${GROUP}" \
        --cluster "slurm" \
        --slurm_gres "gpu:a40:1" \
        --seed $SEED \
        --dict_basis_norm_coeff $DICT_NORM_COEF \
        --dict_basis_cov_coeff $DICT_COV_COEF \
        --dataset_name $DATA \
        --lookback $LOOKBACK \
        --horizon $HORIZON \
        --extend_val True \
        --target_series_index $TARGET
done 

