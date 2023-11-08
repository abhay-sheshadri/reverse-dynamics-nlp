#
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
from utils import create_dataset


#
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

    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
        help='Choose device: cpu or cuda'
    )
    parser.add_argument('--reverse_model_prior', type=bool, default=False,
        help='Use the reverse model as a prior')

    parser.add_argument('--multiple_priors_start_idx', type=int, default=0)
    parser.add_argument('--multiple_priors_end_idx', type=int, default=0)
    parser.add_argument('--model_size', type=str, default='160m')

    return parser.parse_args()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()
    model_sizes = ["70m", "160m", "410m"]
    model_names = ["EleutherAI/pythia-" + size + "-deduped-v0" for size in model_sizes]

    list_of_dataset_names = ["pile_val"] #["small-pile-dedup-train", "TinyStories"]

    for dataset_name in list_of_dataset_names:
        for (model_name, model_size) in zip(model_names, model_sizes):
            model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)
            tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
            tokenizer.pad_token = tokenizer.eos_token

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

            model.eval()
            criterion = torch.nn.CrossEntropyLoss(
                ignore_index=tokenizer.pad_token_id
            )  
            losses = []

            with torch.no_grad():
                for batch in tqdm(dataloader, desc="Computing loss"):
                    input_ids = batch["input_ids"][:, :-1].to(device)
                    targets = batch["input_ids"][:, 1:].to(device)

                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits

                    logits = rearrange(logits, "b n c -> (b n) c")
                    targets = rearrange(targets, "b n -> (b n)")

                    loss = criterion(logits, targets)
                    losses.append(loss.item())

            loss_array = np.array(losses)
            loss_mean = np.mean(loss_array)
            loss_variance = np.var(loss_array)
            nbatches = len(dataloader)

            data = {
                    'name': "forwards-"+model_size,
                    'dataset' : dataset_name,
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

            with open(f"{directory}/forwards-{model_size}-{hash_obj.hexdigest()}.json", 'w') as f:
                json.dump(data, f)

if __name__ == "__main__":
    main()

