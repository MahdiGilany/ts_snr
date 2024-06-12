# deeptime
DATA="etth2" # "exchange_rate" "etth2" "traffic"
LOOKBACK=288 # 288 672
HORIZON=96
DICT_REG_COEF=0.0
TARGET=-1
GROUP="deeptime_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
# GROUP="deeptime_${DICT_REG_COEF}dict-reg_${DATA}_in${LOOKBACK}_out${HORIZON}_vrt${TARGET}"
for SEED in 0   
do
    python deeptime_experiment.py \
        --name "${GROUP}_${SEED}" \
        --group "${GROUP}" \
        --cluster "slurm" \
        --slurm_gres "gpu:a40:1" \
        --seed $SEED \
        --dict_reg_coef $DICT_REG_COEF \
        --dataset_name $DATA \
        --lookback $LOOKBACK \
        --horizon $HORIZON \
        --target_series_index $TARGET
done 

