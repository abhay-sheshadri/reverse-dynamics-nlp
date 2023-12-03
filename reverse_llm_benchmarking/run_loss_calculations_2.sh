#!/bin/bash

prefix_length=29
num_examples_reversal=100
num_examples_models=100
filename_prefix=no_chunk_11_27
dataset_name=pile_val
seed=5491
device=cuda
full_data_set_chunk=false

# Reversal

#python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "160m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
#python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "410m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
#python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior true --model_size "160m" --filename_prefix $filename_prefix  --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
#python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior true --model_size "410m" --filename_prefix $filename_prefix  --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
#python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/probs_30.pt" --multiple_priors_start_idx 0 --multiple_priors_end_idx 29 --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 70m
#python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/probs_30.pt" --multiple_priors_start_idx 0 --multiple_priors_end_idx 29 --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 160m
#python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/probs_30.pt" --multiple_priors_start_idx 0 --multiple_priors_end_idx 29 --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 410m 
#python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/pile_empirical.pt" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 70m
#python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/pile_empirical.pt" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 160m
python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/pythia-70m-deduped-v0_stationary_dist.pt" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 70m
python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 160m
python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/pythia-4100m-deduped-v0_stationary_dist.pt" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 410m
python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/pile_empirical.pt" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 410m



# prefix_length=29
# num_examples_reversal=50
# num_examples_models=1000
# filename_prefix=no_chunk_11_14
# dataset_name=pile_val
# seed=5491
# device=cuda
# full_data_set_chunk=false

# # Reversal

# # python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "160m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# # python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "410m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# # python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/probs_30.pt" --multiple_priors_start_idx 0 --multiple_priors_end_idx 29 --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 70m
# # python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/probs_30.pt" --multiple_priors_start_idx 0 --multiple_priors_end_idx 29 --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 160m
# # python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/probs_30.pt" --multiple_priors_start_idx 0 --multiple_priors_end_idx 29 --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 410m 
# # python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/pile_empirical.pt" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# # python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior true --model_size "410m" --filename_prefix $filename_prefix  --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk




# python reverse_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "70m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "160m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "410m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk

# python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "1B" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk




# prefix_length=29
# num_examples_reversal=50
# num_examples_models=1000
# filename_prefix=no_chunk_11_14
# dataset_name=pile_val
# seed=5491
# device=cuda
# full_data_set_chunk=false

# # Reversal
# python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/probs_30.pt" --multiple_priors_start_idx 0 --multiple_priors_end_idx 29 --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 70m
# python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/probs_30.pt" --multiple_priors_start_idx 0 --multiple_priors_end_idx 29 --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --model_size 160m
# # python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/pile_empirical.pt" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# # python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior true --model_size "410m" --filename_prefix $filename_prefix  --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk

# python reverse_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "70m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# # python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "160m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
# # python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "410m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk

# # python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "1B" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk
