#!/bin/bash
LANGS=(en fr es fi ru id vi zh synth_a synth_b synth_c synth_d)
for i in ${!LANGS[@]}; do
    lang=${LANGS[$i]}
    gpu=$i
    echo "Starting $lang on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu nohup python3 train_exp8_ddp.py $lang > logs/${lang}_training.log 2>&1 &
done
echo 'All training started'
