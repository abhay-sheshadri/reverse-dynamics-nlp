#!/bin/bash
prefix_length=50
num_examples=50
filename_prefix=debug_12_5
dataset_name=pile_val
seed=5491
device=gpu
full_data_set_chunk=false

# Reversal
python all_methods_loss.py --num_examples $num_examples --prefix_length $prefix_length --reverse_model_prior true --model_size "160m" --filename_prefix $filename_prefix  --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
