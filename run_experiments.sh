# deeptime
DATA="exchange_rate" # "exchange_rate" "etth2" "traffic"
LOOKBACK=288 # 288 672
HORIZON=96
DICT_NORM_COEF=0.05
DICT_COV_COEF=1.0
TARGET=-1
# GROUP="deeptime_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# GROUP="deeptime_${DICT_NORM_COEF}dict-norm_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# GROUP="deeptime_${DICT_COV_COEF}dict-cov_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
GROUP="deeptime_${DICT_COV_COEF}dict-cov_${DICT_NORM_COEF}dict-norm_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
for SEED in 0   
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
        --target_series_index $TARGET
done 

