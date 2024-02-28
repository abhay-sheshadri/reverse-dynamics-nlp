#GCG
import os
dir_list = os.chdir('./../reverse-dynamics-nlp/')

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from prompt_optimizer import PromptOptimizer
from utils import get_reverse_pair, start_chunk_hf, forward_loss, reverse_tokenize
from utils import reverse_normalized_generate, reverse_normalized_beam_generate, forward_loss_batch, rand_init
from tqdm import tqdm
import pickle



tokenizer = AutoTokenizer.from_pretrained("afterless/reverse-pythia-160m")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m", cache_dir='/scratch/jp6263/hf/models/').cuda()
tokenizer.eos_token = '<|endoftext|>'
tokenizer.pad_token = tokenizer.eos_token

data = load_dataset('allenai/real-toxicity-prompts', cache_dir='/scratch/jp6263/hf/datasets/')
toxic_stuff = [(d['prompt']['text'],d['continuation']['text']) for d in data['train'] if (d['continuation']['severe_toxicity'] is not None and d['continuation']['severe_toxicity']>0.85)]
print(toxic_stuff[0])
# dataset = load_dataset("NeelNanda/pile-10k")
# pairs = get_reverse_pair(dataset['train'], start_chunk_hf, tokenizer)
# print(next(pairs))
# nanda_list = list(pairs)

all_gcg_losses = []
all_gcg_naturals = []
all_gcg_prefixes = []
eval_size = 250 #45 for 4.5ish hours using 6 weights

for prefix_weight in [0]+[0.02*3**i for i in range(4)]+[1]:
    tokenwise_acc = []
    gcg_loss = []
    gcg_naturals = []
    temp = None #None for default GCG with uniform sampling
    gcg_found_prefixes = []
    gcg = PromptOptimizer(model, tokenizer, n_proposals=128, n_epochs=100, n_top_indices=128, prefix_loss_weight=prefix_weight)

    for p,pair in enumerate(tqdm(toxic_stuff)):
        if len(gcg_loss)==eval_size: break
        prefix, suffix = pair
        prefix_tokens = tokenizer.encode(prefix)
        suffix_tokens = tokenizer.encode(suffix)
        if len(prefix_tokens) > 10: prefix_tokens = prefix_tokens[-10:]
        if len(prefix_tokens) < 10: continue
        # if len(suffix_tokens) < 40: continue
        prefix_loss,suffix_loss = forward_loss(model, pair, tokenizer)
        len_prefix = len(prefix_tokens)
        rand_prefix = rand_init(len_prefix, tokenizer)
        optimized_string = gcg.optimize(rand_prefix, suffix, temperature=temp,)
        predicted_prefix_tokens = tokenizer.encode(optimized_string)[:len_prefix]
        predicted_prefix = tokenizer.decode(predicted_prefix_tokens)
        gcg_found_prefixes.append((p,predicted_prefix))
        predicted_prefix_loss, predicted_suffix_loss = forward_loss(model, (predicted_prefix, suffix), tokenizer)
        print(f'True prefix is:\n{prefix} \n\nPredicted prefix:\n{predicted_prefix}\nfor suffix:\n {suffix}')
        print(f'Loss for suffix given predicted prefix is {predicted_suffix_loss.item()} \n Suffix loss for true prefix is {suffix_loss.item()}')
        print(f'NLL on predicted prefix is {predicted_prefix_loss.item()} \n NLL on true prefix is {prefix_loss.item()}')
        gcg_loss.append(predicted_suffix_loss.item())
        gcg_naturals.append(predicted_prefix_loss.item())
        tokenwise_acc.append(sum([1 for i in range(len(prefix_tokens)) if prefix_tokens[i] == predicted_prefix_tokens[i]])/len(prefix_tokens))
    all_gcg_losses.append(gcg_loss)
    all_gcg_naturals.append(gcg_naturals)
    all_gcg_prefixes.append(gcg_found_prefixes)
    # print(f'Average tokenwise accuracy is {sum(tokenwise_acc)/len(tokenwise_acc)}')
    print(f'Average loss is {sum(gcg_loss)/len(gcg_loss)}')

results_dict = {'gcg_losses':all_gcg_losses, 'gcg_naturals':all_gcg_naturals, 'gcg_prefixes':all_gcg_prefixes}
with open('/home/jp6263/reverse-dynamics-nlp/gcg_results_toxic_250sample.pkl', 'wb') as f:
    pickle.dump(results_dict, f)
    