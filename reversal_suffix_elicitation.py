#reversal
import argparse
import os
dir_list = os.chdir('./../reverse-dynamics-nlp/')

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from prompt_optimizer import PromptOptimizer, ReversalLMPrior, ReverseModelSampler
import pickle
from utils import get_reverse_pair, start_chunk_hf, forward_loss, reverse_tokenize
from utils import reverse_normalized_generate, reverse_normalized_beam_generate, forward_loss_batch, rand_init
from tqdm import tqdm



def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    
    # Common Parameters
    parser.add_argument('--model_size', type = str, default="160m")
    parser.add_argument("--eval_size", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="allenai/real-toxicity-prompts")
    
    
    return parser.parse_args()


def get_statistics(prefix, suffix, optimizer, model, tokenizer):
    # Get prediction according to optimizer
    optimized_string = optimizer.optimize(prefix, suffix)
    optimized_string = optimized_string[:len(optimized_string)-len(suffix)]
    predicted_prefix_loss, predicted_suffix_loss = forward_loss(model, (optimized_string, suffix), tokenizer)
    return optimized_string, predicted_prefix_loss, predicted_suffix_loss


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

    temp = None #None for default reversal with uniform sampling
    
    optimizers = {
        "gcg": PromptOptimizer(model, tokenizer, prefix_loss_weight=0),
        "reverse_model": ReverseModelSampler(reverse_model, tokenizer),
        "bayesian_reversal": ReversalLMPrior(model, reverse_model, tokenizer, batch_size=128)
    }
    
    output_stats = {}
    
    for p, pair in enumerate(tqdm(toxic_stuff[:args.eval_size])):
        
        prefix, suffix = pair
        prefix_tokens = tokenizer.encode(prefix)
        suffix_tokens = tokenizer.encode(suffix)
        
        if len(prefix_tokens) > 10:
            prefix_tokens = prefix_tokens[-10:]
        if len(prefix_tokens) < 10:
            continue
        # if args.dataset == "pile"
        # if len(suffix_tokens) < 40: continue
        prefix_loss, suffix_loss = forward_loss(model, pair, tokenizer)
        
        output_stats[suffix] = {
            "gt_prefix": prefix,
            "gt_prefix_loss": prefix_loss.to("cpu"),
            "gt_suffix_loss": suffix_loss.to("cpu"),
            "prompt_opts": {
                opt: {} for opt in optimizers
            }
        }

        for opt_name, optimizer in optimizers.items():
            
            len_prefix = len(prefix_tokens)
            rand_prefix = rand_init(len_prefix, tokenizer)
            
            optimized_string, predicted_prefix_loss, predicted_suffix_loss = get_statistics(
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
            }

        # all_reversal_losses.append(reversal_loss)
        # all_reversal_naturals.append(reversal_naturals)
        # all_reversal_prefixes.append(reversal_found_prefixes)
        # print(f'Average tokenwise accuracy is {sum(tokenwise_acc)/len(tokenwise_acc)}')
        # print(f'Average loss is {sum(reversal_loss)/len(reversal_loss)}')

    with open(f'data/reversal_results_toxic_{args.model_size}_{args.eval_size}sample.pkl', 'wb') as f:
        pickle.dump(output_stats, f)
        
if __name__ == "__main__":
    main()
