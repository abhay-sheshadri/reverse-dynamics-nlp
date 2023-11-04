#!/usr/bin/python
# -*- coding: utf-8 -*-

# %%

import argparse
import hashlib
import json
import os

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (DataCollatorForLanguageModeling, GPTNeoXForCausalLM,
                          GPTNeoXTokenizerFast)

import stationary_reversal as sr
from reverse_sampling import *
from utils import *

#import sys
#sys.path.append('../stationary_reversal.py')



# The memory usage of this function is dominated by
# the output of the model, (batch_size, sample_length, vocab_size).
# The default values here correspond to 5.24 gigabytes of memory.

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')

    parser.add_argument('--num_examples', type=int, default=10,
        help='Number of examples to run loss over.'
    )

    parser.add_argument('--prefix_length', type=int, default=10,
        help='Number of tokens to predict in each example.'
    )

    parser.add_argument('--suffix_length', type=int, default=1,
        help='Context length for each example.'
    )
    
    parser.add_argument('--num_buffer', type=int, default=0,
        help='Where to begin the prefix.'
    )

    parser.add_argument('--suffix_batch_size', type=int, default=1,
        help='Batch size for loss calculation (i.e. number of suffixes).'
    )

    parser.add_argument('--vocab_batch_size', type=int, default=786,
        help='Number of words to batch when computing reverse probability.'
    )
    
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
        help='Choose device: cpu or cuda'
    )
    
    parser.add_argument('--dist', type=str, required=True,
        help='Path to the distribution file'
    )
    
    parser.add_argument('--dillution', type=float, default=0.0,
        help='dist = (1 - dillution) * dist + dillution * uniform'
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    device = torch.device(args.device)
    if device == 'cuda':
        print('Using gpu.')

    model_sizes = ['160m'] #  ['70m', '160m', '410m']
    model_names = ['EleutherAI/pythia-' + size + '-deduped-v0'
                   for size in model_sizes]

    list_of_dataset_names = ['pile_val']  # ["small-pile-dedup-train", "TinyStories"]

    empirical_dist = torch.load(args.dist)
    uniform_dist = torch.ones_like(empirical_dist) / empirical_dist.shape[0]
    empirical_dist = empirical_dist * (1 - args.dillution) + uniform_dist * args.dillution

    #list_of_stationary_distributions = ['empirical_dist', 'uniform_dist', 'Markov_stationary_dist']

    for dataset_name in list_of_dataset_names:
        
        tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/gpt-neox-20b')

        dataloader = create_dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            num_examples=args.num_examples,
            prefix_length=args.prefix_length,
            suffix_length=args.suffix_length,
            num_buffer=args.num_buffer,
            suffix_batch_size=args.suffix_batch_size,
            seed=42
        )
        
        for (model_name, model_size) in zip(model_names, model_sizes):
            
            model = GPTNeoXForCausalLM.from_pretrained(
                model_name,
                revision="step3000",
                device_map="auto"
            )

            model.eval()
            criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # This is your loss function
            losses = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc='Computing loss'):
                    input_ids = batch['input_ids'].to(device)
                    targets = batch['input_ids'][:, :-1].to(device)

                    print(input_ids.shape)

                    # I assume it is fine to cross entropy with logprobs versus logits it's all the same
                    logits = sr.stationary_reverse_full_dist_suffix_calculation(
                        model,
                        empirical_dist,
                        input_ids,
                        vocab_batch_size=args.vocab_batch_size,
                        renormalize_dist=True
                    )
                    
                    """
                    _, logits = sample_reverse_dynamics(
                        model=model,
                        stationary_dist=empirical_dist,
                        prefix_length=sample_length,
                        tokenized_suffix=input_ids,
                        vocab_batch_size=1024,
                        temperature=0.7,
                        device=device
                    )
                    """

                    # logits = rearrange(logits, 'b n c -> (b n) c')
                    targets = rearrange(targets, 'b n -> (b n)')

                    loss = criterion(logits, targets)
                    losses.append(loss.item())

            loss_array = np.array(losses)
            loss_mean = np.mean(loss_array)
            loss_variance = np.var(loss_array)
            nbatches = len(dataloader)

            data = {
                'mean': loss_mean,
                'variance': loss_variance,
                'std_on_mean': np.std(loss_array) / np.sqrt(nbatches),
                'nbatches': nbatches,
            }
            args_dict = vars(args)
            data.update(args_dict)

            directory = 'data/' + dataset_name
            if not os.path.exists(directory):
                os.makedirs(directory)
                
            dict_str = json.dumps(data, sort_keys=True)
            hash_obj = hashlib.md5(dict_str.encode())

            with open(f"{directory}/stationary-reversal-{model_size}-{hash_obj.hexdigest()}.json", 'w') as f:
                json.dump(data, f)


      # np.save(directory+"/stationary-reversal-" + model_size + "-loss-samples.npy", loss_array)

if __name__ == '__main__':
    main()
