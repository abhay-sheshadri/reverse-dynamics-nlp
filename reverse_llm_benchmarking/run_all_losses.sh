#!/bin/bash
prefix_length=19
suffix_length=1
num_examples=100
filename_prefix=al_12_6_
dataset_name=pile_val
seed=5491
device=cuda
full_data_set_chunk=false
vocab_batch_size=500

# Reversal
python all_methods_loss.py --num_examples $num_examples --prefix_length $prefix_length --reverse_model_prior true --model_size "160m" --filename_prefix $filename_prefix  --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --vocab_batch_size $vocab_batch_size
