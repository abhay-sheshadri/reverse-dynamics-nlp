#!/bin/bash

# Comparing LMs
# prefix_length=10
# num_examples_reversal=2
# num_examples_models=2
# filename_prefix=test_run_
# dataset_name=DebugDataSet
# seed=5491
# device=cpu


# python reverse_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device
# python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "160m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device
# python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "410m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device
# python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "1B" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device


# With Reversal
prefix_length=29
num_examples_reversal=2
num_examples_models=2
filename_prefix=test_run_
dataset_name=DebugDataSet
seed=5491
device=cpu

# Reversal
python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/probs_30.pt" --multiple_prior_start_idx 0 --multiple_prior_end_idx 28 --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device
# python reverse_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device
# python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "160m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device
# #python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "410m" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device
# #python forward_llm_loss.py --num_examples $num_examples_models --prefix_length $prefix_length --model_size "1B" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device
# python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior false --dist "../data/distributions/pile_empirical.pt" --filename_prefix $filename_prefix --dataset_name $dataset_name --seed $seed --device $device
# python reversal_loss.py --num_examples $num_examples_reversal --prefix_length $prefix_length --reverse_model_prior true --model_size "410m" --filename_prefix $filename_prefix  --dataset_name $dataset_name --seed $seed --device $device







# python reverse_llm_loss.py --num_examples 100 --prefix_length 50
# python forward_llm_loss.py --num_examples 100 --prefix_length 50

# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --reverse_model_prior true 
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --reverse_model_prior true --dilution 0.2
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --reverse_model_prior true --dilution 0.4
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --reverse_model_prior true --dilution 0.6

# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --reverse_model_prior true --dilution --model_size 410m


# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.0 --dist ../data/distributions/probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.2 --dist ../data/distributions/probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.4 --dist ../data/distributions/probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9

# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.0 --dist ../data/distributions/smoothed_probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.2 --dist ../data/distributions/smoothed_probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 0 --dilution 0.4 --dist ../data/distributions/smoothed_probs_10.pt --multiple_priors_start_idx 0 --multiple_priors_end_idx 9



# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --dist ../data/distributions/pile10k_empirical.pt
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.2 --dist ../data/distributions/pile10k_empirical.pt
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.4 --dist ../data/distributions/pile10k_empirical.pt
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt

# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.2 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.4 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 1.0 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt

# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 10 --dist ../data/distributions/pile10k_empirical.pt
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 20 --dist ../data/distributions/pile10k_empirical.pt

# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 10 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt
# python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --num_buffer 20 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt






