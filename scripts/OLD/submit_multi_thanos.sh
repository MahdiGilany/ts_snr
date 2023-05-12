conda activate exact 

#for split_seed in {0..3}

#do
split_seed=3
for seed in {0..3}
do 
    export SPLIT_SEED=$split_seed
    export SEED=$seed
    python main.py experiment=01_pretrain_tuffc_PWilson_2023-01-09
done 
#done 