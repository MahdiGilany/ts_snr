# deeptime
DATA="exchange_rate" # "exchange_rate" "etth2" "traffic"
LOOKBACK=288
HORIZON=96
DICT_REG_COEF=0.05
# GROUP="deeptime_${DATA}_in${LOOKBACK}_out${HORIZON}"
GROUP="deeptime_.05dict-reg_${DATA}_in${LOOKBACK}_out${HORIZON}"
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
        --horizon $HORIZON
done 

