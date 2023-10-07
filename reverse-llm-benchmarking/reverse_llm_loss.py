#
import argparse
from tqdm import tqdm
from transformers import (
    GPTNeoXForCausalLM,
    GPTNeoXTokenizerFast,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from einops import rearrange
from datasets import concatenate_datasets
import numpy as np
import json
import os

#


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some arguments.")

    parser.add_argument(
        "--samples", type=int, default=10000, help="Number of samples to keep."
    )

    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for training."
    )

    return parser.parse_args()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()
    sample_size = args.samples
    batch_size = args.batch_size
    #
    reverse_model = GPTNeoXForCausalLM.from_pretrained(
        "afterless/reverse-pythia-160m"
    ).to(device)

    #
    # dataset = load_dataset('NeelNanda/pile-10k')
    # dataset = dataset.map(lambda examples: {"text": examples['text'][::-1]}, batched=True)

    # Load the test split of the Pile dataset
    # dataset = load_dataset("ola13/small-the_pile-dedup", split="test")

    list_of_dataset_names = ["pile_val"] #["small-pile-dedup-train", "TinyStories"]
    for dataset_name in list_of_dataset_names:
        if dataset_name == "small-pile-dedup-train":
            #
            # Using the Pile
            dataset = load_dataset("ola13/small-the_pile-dedup")
            # Concatenate all datasets in the DatasetDict
            concatenated_dataset = concatenate_datasets([ds for ds in dataset.values()])
            # Shuffle the concatenated dataset
            shuffled_dataset = concatenated_dataset.shuffle(
                seed=42
            )  # You can set your desired seed

            # Take a sample. For example, if you want a sample of 10,000 rows:
            sampled_dataset = shuffled_dataset.select(range(sample_size))
            dataset = sampled_dataset
        elif dataset_name == "TinyStories":
            # Using the Tiny Stories Data Set
            dataset = load_dataset("roneneldan/TinyStories", split="validation")
            dataset = dataset.select(range(sample_size))
        elif dataset_name == "pile_val":
            dataset = load_dataset("json", data_files="val.jsonl")
            dataset = dataset["train"].select(range(sample_size))

        #
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

        #

        tokenizer.pad_token = tokenizer.eos_token

        def reverse_text(example):
            return {
                "input_ids": tokenizer.encode(
                    example["text"][::-1],
                    truncation=True,
                    padding="max_length",
                    max_length=2048,
                    return_tensors="pt",
                ).squeeze(0)
            }

        tokenized_dataset = dataset.map(reverse_text)

        all_columns = tokenized_dataset.column_names

        columns_to_remove = [column for column in all_columns if column != "input_ids"]

        tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
        # test reversal: tokenizer.decode(tokenized_dataset["input_ids"][0])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False, return_tensors="pt"
        )

        dataloader = DataLoader(
            tokenized_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size,
        )

        reverse_model.eval()
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        losses = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing loss"):
                input_ids = batch["input_ids"][:, :-1].to(device)
                targets = batch["input_ids"][:, 1:].to(device)

                outputs = reverse_model(input_ids=input_ids)
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
            "mean": loss_mean,
            "variance": loss_variance,
            "std_on_mean": np.std(loss_array) / np.sqrt(nbatches),
            "total_samples": sample_size,
            "batch_size": batch_size,
            "nbatches": nbatches,
        }

        directory = "data/" + dataset_name
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(directory + "/reverse-llm-loss.json", "w") as f:
            json.dump(data, f)

        # np.save(directory+"/reverse-loss-samples.npy", loss_array)


if __name__ == "__main__":
    main()
