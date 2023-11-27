#!/usr/bin/python
# -*- coding: utf-8 -*-

# %%

import argparse
import pickle
import os

import numpy as np
import torch
from tqdm import tqdm
import time
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast

import stationary_reversal as sr
from reverse_sampling import *
from utils import create_dataset, create_chunked_dataset_from_full_sequences,str2bool


# The memory usage of this function is dominated by
# the output of the model, (batch_size, sample_length, vocab_size).
# The default values here correspond to 5.24 gigabytes of memory.

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')

    # Common Parameters
    parser.add_argument('--dataset_name', type=str, default='pile_val')
    parser.add_argument('--num_examples', type=int, default=10,
        help='Number of examples to run loss over.'
    )
    parser.add_argument('--full_data_set_chunk', type=str2bool, default=True)
    parser.add_argument('--prefix_length', type=int, default=10,
        help='Number of tokens to predict in each example.'
    )
    parser.add_argument('--suffix_length', type=int, default=1,
        help='Context length for each example.'
    )
    parser.add_argument('--num_buffer', type=int, default=0,
        help='Where to begin the prefix.'
    )
    parser.add_argument('--batch_size', type=int, default=1,
        help='Batch size for loss calculation (i.e. number of suffixes).'
    )
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed.'
    )
    parser.add_argument('--filename_prefix', type=str, default="")
    parser.add_argument('--return_all_sequences', type=str2bool, default=False)
    parser.add_argument('--filter_small_sequences', type=str2bool, default=False)


    # Reversal Parameters    
    parser.add_argument('--vocab_batch_size', type=int, default=786,
        help='Number of words to batch when computing reverse probability.'
    )    
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
        help='Choose device: cpu or cuda'
    )    
    parser.add_argument('--dist', type=str, required=False,
        help='Path to the distribution file'
    )
    parser.add_argument('--dilution', type=float, default=0.0,
        help='dist = (1 - dilution) * dist + dilution * uniform'
    )
    parser.add_argument('--reverse_model_prior', type=str2bool, default=False,
        help='Use the reverse model as a prior')
    parser.add_argument('--multiple_priors_start_idx', type=int, default=0)
    parser.add_argument('--multiple_priors_end_idx', type=int, default=0)
    parser.add_argument('--model_size', type=str, default='160m')


    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.full_data_set_chunk:
        print("using chunk full")
    else:
        print("Using fixed windows.")

    if args.reverse_model_prior:
        print("using reverse model prior")
    else:
        print("not using reverse model prior")

    device = torch.device(args.device)
    if device == 'cuda':
        print('Using gpu.')

    model_name = 'EleutherAI/pythia-' + args.model_size + '-deduped'

    if args.reverse_model_prior:
        reverse_model = GPTNeoXForCausalLM.from_pretrained(
            "afterless/reverse-pythia-160m"
        ).to(device)
    else:
        empirical_dist = torch.load(args.dist)
        if args.multiple_priors_end_idx > 0:
            empirical_dist = empirical_dist[:,args.multiple_priors_start_idx:args.multiple_priors_end_idx]
    
    tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')

    if args.full_data_set_chunk:
        dataloader = create_chunked_dataset_from_full_sequences(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            num_examples=args.num_examples,
            prefix_length=args.prefix_length,
            suffix_length=args.suffix_length,
            batch_size=args.batch_size,
            seed=args.seed,
            return_all=args.return_all_sequences,
            filter_small_sequences=args.filter_small_sequences
        )
    else:
        dataloader = create_dataset(
            dataset_name=args.dataset_name,
            tokenizer=tokenizer,
            num_examples=args.num_examples,
            prefix_length=args.prefix_length,
            suffix_length=args.suffix_length,
            num_buffer=args.num_buffer,
            batch_size=args.batch_size,
            seed=args.seed
        )
        
    model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)

    model.eval()
    times = []
    prefixes = []
    suffixes = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Computing loss'):
            # When the dataset is chunked, the leftover piece is kept. 
            # However, sometimes the leftover piece is of size 1, and should be 
            # skipped. 
            if batch["input_ids"].shape[1] == 1:
                continue
            input_ids = batch['input_ids'].to(device)

            if args.reverse_model_prior:
                start_time = time.time()
                out = sample_reverse_dynamics_reverse_prior(
                    model,
                    reverse_model,
                    args.prefix_length,
                    input_ids[-args.suffix_length:],
                    vocab_batch_size=args.vocab_batch_size,
                    temperature=1.0,
                    device=device
                )
                end_time = time.time()
            else:
                start_time = time.time()
                out = sample_reverse_dynamics(
                    model,
                    empirical_dist,
                    args.prefix_length,
                    input_ids[-args.suffix_length:],
                    vocab_batch_size=args.vocab_batch_size,
                    temperature=1.0,
                    device=device
                )
                end_time = time.time()
            times.append(end_time - start_time)
            prefixes.append(out[0][0:args.prefix_length])
            suffixes.append(out[0][args.prefix_length:])
                                                        
    time_array = np.array(times)
    time_mean = np.mean(time_array)
    time_variance = np.var(time_array)
    ntimes = len(times)

    data = {
        'name': "stationary_reversal",
        'dataset' : args.dataset_name,
        'time_mean': time_mean,
        'time_variance': time_variance,
        'time_std_on_mean': np.std(time_array) / np.sqrt(ntimes),
        'ntimes': ntimes,
    }
    args_dict = vars(args)
    data.update(args_dict)
    data["times"] = times
    data["prefixes"] = prefixes
    data["suffixes"] = suffixes

    directory = 'data/timing/' + args.dataset_name
    if not os.path.exists(directory):
        os.makedirs(directory)


    with open(f"{directory}/{args.filename_prefix}reversal-timing-{args.model_size}-reversior-prior-{args.reverse_model_prior}.pkl", 'w') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()

