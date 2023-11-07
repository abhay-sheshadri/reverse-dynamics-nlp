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
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument(
        "--samples", type=int, default=10000, help="Number of samples to keep."
    )

    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for training."
    )

    parser.add_argument('--sample_length', type=int, default=2048,
                      help='Where to truncate the input sequences.'
    )

    return parser.parse_args()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()
    sample_size = args.samples
    batch_size = args.batch_size
    sample_length = args.sample_length
    #
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

            with open(f"{directory}/forwards-model-{model_size}-{hash_obj.hexdigest()}.json", 'w') as f:
                json.dump(data, f)

if __name__ == "__main__":
    main()
    
