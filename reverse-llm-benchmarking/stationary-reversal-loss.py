
#%%
import argparse
from tqdm import tqdm
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from einops import rearrange
from datasets import concatenate_datasets
import numpy as np
import json
import os

import sys
sys.path.append("../stationary_reversal.py")

import stationary_reversal as sr 

#%%
def parse_arguments():
  parser = argparse.ArgumentParser(description="Process some arguments.")
  
  parser.add_argument('--samples', type=int, default=10000, 
                      help='Number of samples to keep.')
  
  parser.add_argument('--batch_size', type=int, default=10,
                      help='Batch size for loss calculation.')
  
  parser.add_argument('--sample_length', type=int, default=2048,
                      help='Where to truncate the input sequences.')
  



  return parser.parse_args()


def main():

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  args = parse_arguments()
  sample_size = args.samples
  batch_size = args.batch_size
  sample_length = args.sample_length
  #%%
  model_sizes = ["70m", "160m", "410m"]
  model_names = ["EleutherAI/pythia-" + size + "-deduped-v0" for size in model_sizes]

  list_of_dataset_names = ["pile_val"]  # ["small-pile-dedup-train", "TinyStories"]
  
  empirical_dist = torch.load("../data/pile10k_empirical.pt")
  
  for dataset_name in list_of_dataset_names:
    if dataset_name == "small-pile-dedup-train":
        # Using the Pile
        dataset = load_dataset(
        "ola13/small-the_pile-dedup"
        )
        # Concatenate all datasets in the DatasetDict
        concatenated_dataset = concatenate_datasets([ds for ds in dataset.values()])  
        # Shuffle the concatenated dataset
        shuffled_dataset = concatenated_dataset.shuffle(seed=42)  # You can set your desired seed

        sampled_dataset = shuffled_dataset.select(range(sample_size))
        dataset = sampled_dataset
    elif dataset_name == "TinyStories":
      # Using the Tiny Stories Data Set
      dataset = load_dataset("roneneldan/TinyStories", split="validation")
      dataset = dataset.select(range(sample_size))
    elif dataset_name == "pile_val":
      dataset = load_dataset("json", data_files="data/val.jsonl")
      dataset = dataset["train"].select(range(sample_size))

    for (model_name, model_size) in zip(model_names, model_sizes):
      model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)
    
      tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

      tokenizer.pad_token = tokenizer.eos_token    
      def tokenize_text(example):
        return {
          "input_ids": tokenizer.encode(
            example["text"], 
            truncation=True, 
            padding='max_length', 
            max_length=sample_length, 
            return_tensors="pt"
          ).squeeze(0)
        }

      tokenized_dataset = dataset.map(tokenize_text)

      # Get all column names
      all_columns = tokenized_dataset.column_names

      # Find columns to remove
      columns_to_remove = [column for column in all_columns if column != "input_ids"]

      # Remove unwanted columns
      tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

      # Debug
      # first_array = np.array(tokenized_dataset["input_ids"])
      # # max_occuring_token = max(tokenized_dataset["input_ids"], key=tokenized_dataset["input_ids"].count)
      # num_zeros = np.count_nonzero(first_array == 0)
      # num_nonzeros = np.count_nonzero(first_array != 0)
      # print(num_zeros/(num_zeros + num_nonzeros))

      # Use DataCollator to handle padding during training
      data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt")

      # Convert dataset to DataLoader for batch processing
      dataloader = DataLoader(
        tokenized_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=batch_size)  

      model.eval()  
      criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)  # This is your loss function
      losses = []

      with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing loss"):
          input_ids = batch["input_ids"][:, :-1].to(device)  
          targets = batch["input_ids"][:, 1:].to(device) 

          print(input_ids.shape)

          # I assume it is fine to cross entropy with logprobs versus logits it's all the same
          outputs = sr.stationary_reverse_full_dist_suffix_calculation(
            model, empirical_dist, input_ids
          )

          logits = rearrange(logits, 'b n c -> (b n) c')
          targets = rearrange(targets, 'b n -> (b n)')
          
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
        "nbatches": nbatches
      }

      directory = "data/" + dataset_name
      if not os.path.exists(directory):
        os.makedirs(directory)

      with open(directory+"/stationary-reversal-" + model_size+ "-loss.json", "w") as f:
        json.dump(data, f)


      # np.save(directory+"/stationary-reversal-" + model_size + "-loss-samples.npy", loss_array)

if __name__ == "__main__":
  main()
