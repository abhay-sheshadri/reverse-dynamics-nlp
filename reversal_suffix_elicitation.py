#reversal
import os
dir_list = os.chdir('./../reverse-dynamics-nlp/')

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from prompt_optimizer import PromptOptimizer, ReversalLMPrior
from utils import get_reverse_pair, start_chunk_hf, forward_loss, reverse_tokenize
from utils import reverse_normalized_generate, reverse_normalized_beam_generate, forward_loss_batch, rand_init
from tqdm import tqdm
import pickle


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some arguments.')
    
    # Common Parameters
    parser.add_argument('--num_params', type=int, default=160)
    parser.add_argument("--eval_size", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="allenai/real-toxicity-prompts")
    
    return parser.parse_args()


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained("afterless/reverse-pythia-160m")
    model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/pythia-{args.num_params}m").cuda()
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

    all_reversal_losses = []
    all_reversal_naturals = []
    all_reversal_prefixes = []

    for prefix_weight in [0]+[0.02*3**i for i in range(4)]+[1]:
        tokenwise_acc = []
        reversal_loss = []
        reversal_naturals = []
        temp = None #None for default reversal with uniform sampling
        reversal_found_prefixes = []
        reversal = ReversalLMPrior(model, reverse_model, tokenizer, batch_size=2000)

        for p,pair in enumerate(tqdm(toxic_stuff)):
            if len(reversal_loss)==args.eval_size: break
            prefix, suffix = pair
            prefix_tokens = tokenizer.encode(prefix)
            suffix_tokens = tokenizer.encode(suffix)
            if len(prefix_tokens) > 10: prefix_tokens = prefix_tokens[-10:]
            if len(prefix_tokens) < 10: continue
            # if len(suffix_tokens) < 40: continue
            prefix_loss,suffix_loss = forward_loss(model, pair, tokenizer)
            len_prefix = len(prefix_tokens)
            rand_prefix = rand_init(len_prefix, tokenizer)
            optimized_string = reversal.optimize(rand_prefix, suffix, temperature=temp,)
            predicted_prefix_tokens = tokenizer.encode(optimized_string)[:len_prefix]
            predicted_prefix = tokenizer.decode(predicted_prefix_tokens)
            reversal_found_prefixes.append((p,predicted_prefix))
            predicted_prefix_loss, predicted_suffix_loss = forward_loss(model, (predicted_prefix, suffix), tokenizer)
            print(f'True prefix is:\n{prefix} \n\nPredicted prefix:\n{predicted_prefix}\nfor suffix:\n {suffix}')
            print(f'Loss for suffix given predicted prefix is {predicted_suffix_loss.item()} \n Suffix loss for true prefix is {suffix_loss.item()}')
            print(f'NLL on predicted prefix is {predicted_prefix_loss.item()} \n NLL on true prefix is {prefix_loss.item()}')
            reversal_loss.append(predicted_suffix_loss.item())
            reversal_naturals.append(predicted_prefix_loss.item())
            tokenwise_acc.append(sum([1 for i in range(len(prefix_tokens)) if prefix_tokens[i] == predicted_prefix_tokens[i]])/len(prefix_tokens))
        all_reversal_losses.append(reversal_loss)
        all_reversal_naturals.append(reversal_naturals)
        all_reversal_prefixes.append(reversal_found_prefixes)
        # print(f'Average tokenwise accuracy is {sum(tokenwise_acc)/len(tokenwise_acc)}')
        print(f'Average loss is {sum(reversal_loss)/len(reversal_loss)}')

    results_dict = {'reversal_losses':all_reversal_losses, 'reversal_naturals':all_reversal_naturals, 'reversal_prefixes':all_reversal_prefixes}
    with open(f'/data/reversal_results_toxic_{args.num_params}_{args.eval_size}sample.pkl', 'wb') as f:
        pickle.dump(results_dict, f) 
        
if __name__ == "__main__":
    main()