#!/bin/bash
prefix_length=10
suffix_length=10
num_examples=50
filename_prefix=run_11_26
dataset_name=pile_val
seed=5491
vocab_batch_size=1000

python time_reversal.py --model_size 160m --dist ../data/distributions/pile_empirical.pt --vocab_batch_size $vocab_batch_size --suffix_length $suffix_length --prefix_length $prefix_length --num_examples $num_examples --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed
python time_reversal.py --model_size 410m --dist ../data/distributions/pile_empirical.pt --vocab_batch_size $vocab_batch_size --suffix_length $suffix_length --prefix_length $prefix_length --num_examples $num_examples --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed
python time_reversal.py --model_size 1B --dist ../data/distributions/pile_empirical.pt --vocab_batch_size $vocab_batch_size --suffix_length $suffix_length --prefix_length $prefix_length --num_examples $num_examples --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed
python time_reversal.py --model_size 410m --reverse_model_prior true --vocab_batch_size $vocab_batch_size --suffix_length $suffix_length --prefix_length $prefix_length --num_examples $num_examples --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed
python time_reversal.py --model_size 1B --reverse_model_prior true --vocab_batch_size $vocab_batch_size --suffix_length $suffix_length --prefix_length $prefix_length --num_examples $num_examples --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed
