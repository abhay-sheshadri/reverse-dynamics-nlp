#reversal
import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from src import *
import pickle

import time
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    
    # Common Parameters
    parser.add_argument('--model_size', type = str, default="160m")
    parser.add_argument("--eval_size", type=int, default=50)
    parser.add_argument("--dataset", type=str, required=True, choices=[
        "allenai/real-toxicity-prompts", "NeelNanda/pile-10k", "pile_val"
    ])
    parser.add_argument("--num_prefix_tokens", type=int, default=10)
    parser.add_argument("--num_suffix_tokens", type=int, default=40)
    parser.add_argument("--reversal_num_tokens", type=int, default=10000)
    parser.add_argument("--vocab_batch_size", type=int, default=1000)
    parser.add_argument("--BoN", type=int, default=5)#5
    parser.add_argument("--beam_width", type=int, default=50)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--filename_prefix", type=str, default="")
    parser.add_argument("--optimizers", type=str, nargs='+', default=["gcg", "reverse_model", "bayesian_reversal", "bayesian_bon"])
    parser.add_argument("--write_dir", type=str, default="data")
    
    return parser.parse_args()


def get_statistics(prefix, suffix, optimizer, model, tokenizer):
    # Get prediction according to optimizer
    t1 = time.time()
    optimized_string = optimizer.optimize(prefix, suffix)
    optimized_string = optimized_string[:len(optimized_string)-len(suffix)]
    t2 = time.time()
    predicted_prefix_loss, predicted_suffix_loss = forward_loss(model, (optimized_string, suffix), tokenizer)
    return optimized_string, predicted_prefix_loss, predicted_suffix_loss, t2-t1


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained("afterless/reverse-pythia-160m")
    if "ON_GREENE" in os.environ.keys():
        model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/pythia-{args.model_size}-deduped",cache_dir="/scratch/jp6263/hf/models/").cuda()
    else:
        model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/pythia-{args.model_size}-deduped").cuda()

    reverse_model = GPTNeoXForCausalLM.from_pretrained("afterless/reverse-pythia-160m").cuda()
    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == "allenai/real-toxicity-prompts":
        data = load_dataset(args.dataset)
        ps_pairs = [(d['prompt']['text'],d['continuation']['text']) for d in data['train'] if (d['continuation']['severe_toxicity'] is not None and d['continuation']['severe_toxicity']>0.85)]
        print(ps_pairs[0])
        dataset_name = "real_toxicity_prompts"
    elif args.dataset == "NeelNanda/pile-10k":
        data = load_dataset(args.dataset)
        pairs = get_reverse_pair(data['train'], lambda x1,x2: start_chunk_hf(x1, x2, num_prefix_tokens=args.num_prefix_tokens, num_suffix_tokens = args.num_suffix_tokens), tokenizer)
        print(next(pairs))
        ps_pairs = list(pairs)
        dataset_name = "pile-10k"
    elif args.dataset == "pile_val":
        data = load_dataset('json', data_files='data/val.jsonl')
        pairs = get_reverse_pair(data['train'], lambda x1,x2: start_chunk_hf(x1, x2, num_prefix_tokens=args.num_prefix_tokens, num_suffix_tokens = args.num_suffix_tokens), tokenizer)
        print(next(pairs))
        ps_pairs = list(pairs)
        dataset_name = "pile_val"


    temp = None #None for default reversal with uniform sampling
    
    all_optimizers = {
        "gcg": GreedyCoordinateGradient(model, tokenizer, prefix_loss_weight=args.lam,),
        "reverse_model": ReverseModelHFBeamSearch(model, reverse_model, tokenizer, num_beams=args.beam_width),
        "bayesian_reversal": ReversalLMPrior(model, reverse_model, tokenizer, batch_size=args.vocab_batch_size, num_top_tokens=args.reversal_num_tokens),
        "bayesian_bon": ReversalLMPriorBoN(model, reverse_model, tokenizer, batch_size=args.vocab_batch_size, num_top_tokens=args.reversal_num_tokens, N=args.BoN,)
    }
    optimizers = {k: v for k, v in all_optimizers.items() if k in args.optimizers}
    
    output_stats = {}
    output_stats["parameters"] = {}
    args_dict = vars(args)
    output_stats["parameters"].update(args_dict)
    
    for p, pair in enumerate(tqdm(ps_pairs)):
        if len(output_stats)>=args.eval_size:
            break
        prefix, suffix = pair
        prefix_tokens = tokenizer.encode(prefix)
        suffix_tokens = tokenizer.encode(suffix)
        
        if len(prefix_tokens) > args.num_prefix_tokens:
            prefix_tokens = prefix_tokens[-args.num_prefix_tokens:]
        if len(prefix_tokens) < args.num_prefix_tokens:
            continue
        if len(suffix_tokens) < args.num_suffix_tokens and args.dataset != "allenai/real-toxicity-prompts": 
            continue
        prefix_loss, suffix_loss = forward_loss(model, pair, tokenizer)
        
        output_stats[suffix] = {
            "gt_prefix": prefix,
            "gt_prefix_loss": prefix_loss.item(),
            "gt_suffix_loss": suffix_loss.item(),
            "prompt_opts": {
                opt: {} for opt in optimizers
            }
        }

        for opt_name, optimizer in optimizers.items():
            
            len_prefix = len(prefix_tokens)
            rand_prefix = rand_init(len_prefix, tokenizer)
            
            optimized_string, predicted_prefix_loss, predicted_suffix_loss, dt = get_statistics(
                rand_prefix,
                suffix,
                optimizer,
                model,
                tokenizer
            )
            
            output_stats[suffix]["prompt_opts"][opt_name] = {
                "prefix": optimized_string,
                "prefix_loss": predicted_prefix_loss.item(),
                "suffix_loss": predicted_suffix_loss.item(),
                "time": dt
            }
            print("method: ", opt_name, "time: ", dt, "suffix_loss:", predicted_suffix_loss.item())

        if p in [args.eval_size//10,args.eval_size//2]:
            with open(args.write_dir+f'/{args.filename_prefix}temp_{p}_reversal_results_{dataset_name}_{args.model_size}_{args.eval_size}sample.pkl', 'wb') as f:
                pickle.dump(output_stats, f)

    with open(args.write_dir+f'/{args.filename_prefix}reversal_results_{dataset_name}_{args.model_size}_{args.eval_size}sample.pkl', 'wb') as f:
        pickle.dump(output_stats, f)
        
if __name__ == "__main__":
    main()
