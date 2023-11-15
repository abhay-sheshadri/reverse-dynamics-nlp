#reversal
import argparse
import os
dir_list = os.chdir('./../reverse-dynamics-nlp/')

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from prompt_optimizer import PromptOptimizer, ReversalLMPrior, ReverseModelSampler
from utils import get_reverse_pair, start_chunk_hf, forward_loss, reverse_tokenize
from utils import reverse_normalized_generate, reverse_normalized_beam_generate, forward_loss_batch, rand_init
from tqdm import tqdm
import pickle

import time


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    
    # Common Parameters
    parser.add_argument('--model_size', type = str, default="160m")
    parser.add_argument("--eval_size", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="allenai/real-toxicity-prompts")
    
    parser.add_argument("--vocab_batch_size", type=int, default=1024)
    
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained("afterless/reverse-pythia-160m")
    model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/pythia-{args.model_size}-deduped").cuda()
    reverse_model = GPTNeoXForCausalLM.from_pretrained("afterless/reverse-pythia-160m").cuda()
    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.pad_token = tokenizer.eos_token

    data = load_dataset(args.dataset)
    toxic_stuff = [(d['prompt']['text'],d['continuation']['text']) for d in data['train'] if (d['continuation']['severe_toxicity'] is not None and d['continuation']['severe_toxicity']>0.85)]
    print(toxic_stuff[0])
    # dataset = load_dataset("NeelNanda/pile-10k")
    # pairs = get_reverse_pair(dataset['train'], start_chunk_hf, tokenizer)
    # print(next(pairs))
    # nanda_list = list(pairs)

    reversal_times = []
    rms_times = []
    
    reversal = ReversalLMPrior(model, reverse_model, tokenizer, batch_size=args.vocab_batch_size)
    rms = ReverseModelSampler(reverse_model, tokenizer)

    for p,pair in enumerate(tqdm(toxic_stuff)):
        if len(reversal_times)==args.eval_size: break
        prefix, suffix = pair
        prefix_tokens = tokenizer.encode(prefix)
        suffix_tokens = tokenizer.encode(suffix)
        if len(prefix_tokens) > 10: prefix_tokens = prefix_tokens[-10:]
        if len(prefix_tokens) < 10: continue
        # if args.dataset == "pile"
        # if len(suffix_tokens) < 40: continue
        len_prefix = len(prefix_tokens)
        rand_prefix = rand_init(len_prefix, tokenizer)

        start_time = time.time()
        reversal_optimized_string = reversal.optimize(rand_prefix, suffix, temperature=0.7,)
        end_time = time.time()
        reversal_time = end_time - start_time

        # For rms_optimized_string
        start_time = time.time()
        rms_optimized_string = rms.optimize(rand_prefix, suffix, temperature=0.7,)
        end_time = time.time()
        rms_time = end_time - start_time

        # Display RMS optimization time in minutes and seconds
        reversal_times.append(reversal_time)
        rms_times.append(rms_time)

    results_dict = {'reversal_times': reversal_times, 'reverse_model_sampling_times':rms_times}
    with open(f'data/time_benchmark_toxic_{args.model_size}_{args.eval_size}sample.pkl', 'wb') as f:
        pickle.dump(results_dict, f) 
        
if __name__ == "__main__":
    main()
