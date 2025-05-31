#!/bin/bash

set -e

workdir='.'
model_name='ours'
ckpt_name='cut3r_512_dpt_4_64'
model_weights="${workdir}/src/${ckpt_name}.pth"
data='7scenes'


output_dir="${workdir}/eval_results/relpose/${data}_${model_name}"
echo "$output_dir"
accelerate launch --num_processes 8 --main_process_port 29558 eval/relpose/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --size 512



