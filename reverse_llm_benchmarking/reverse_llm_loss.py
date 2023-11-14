#
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from utils import create_dataset, create_chunked_dataset_from_full_sequences, str2bool


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
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
        help='Choose device: cpu or cuda'
    )
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed.'
    )
    parser.add_argument('--filename_prefix', type=str, default="")
    parser.add_argument('--return_all_sequences', type=str2bool, default=False)
    parser.add_argument('--filter_small_sequences', type=str2bool, default=False)

    return parser.parse_args()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()

    reverse_model = GPTNeoXForCausalLM.from_pretrained(
        "afterless/reverse-pythia-160m"
    ).to(device)


    tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token
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

    reverse_model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing loss"):
            # When the dataset is chunked, the leftover piece is kept. 
            # However, sometimes the leftover piece is of size 1, and should be 
            # skipped. 
            if batch["input_ids"].shape[1] == 1:
                continue
            reversed_input_ids = batch["input_ids"].flip(dims=[1]).to(device)        
            
            input_ids = reversed_input_ids[:, :-1]
            targets = reversed_input_ids[:, 1:]


            outputs = reverse_model(input_ids=input_ids)
            logits = outputs.logits

            logits = rearrange(logits, "b n c -> (b n) c")
            targets = rearrange(targets, "b n -> (b n)")

            loss = criterion(logits, targets)
            losses.append(loss.item())

    loss_array = np.array(losses)
    loss_mean = np.mean(loss_array)
    loss_variance = np.var(loss_array)
    nlosses = len(losses)

    data = {
        'name': "reverse-160m",
        'dataset' : args.dataset_name,
        'mean': loss_mean,
        'variance': loss_variance,
        'std_on_mean': np.std(loss_array) / np.sqrt(nlosses),
        'nlosses': nlosses,
    }
    args_dict = vars(args)
    data.update(args_dict)

    directory = 'data/' + args.dataset_name
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    dict_str = json.dumps(data, sort_keys=True)
    hash_obj = hashlib.md5(dict_str.encode())

    with open(f"{directory}/{args.filename_prefix}reverse-160m-{hash_obj.hexdigest()}.json", 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    main()
