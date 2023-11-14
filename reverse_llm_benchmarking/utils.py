import argparse
from datasets import concatenate_datasets, load_dataset
import random
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling


def create_dataset(
    dataset_name,
    tokenizer,
    num_examples,
    prefix_length,
    suffix_length=1,
    num_buffer=0,
    batch_size=1,
    seed=42
):
          
    # Check that the dataset is in the list of valid datasets
    list_of_dataset_names = ["pile_val", "small-pile-dedup-train", "TinyStories", "DebugDataSet"]
    assert dataset_name in list_of_dataset_names
    
    if dataset_name == 'small-pile-dedup-train':
        # Using the Pile
        dataset = load_dataset('ola13/small-the_pile-dedup')
        # Concatenate all datasets in the DatasetDict
        concatenated_dataset = concatenate_datasets([ds for ds in dataset.values()])
        # Shuffle the concatenated dataset
        shuffled_dataset = concatenated_dataset.shuffle(seed=42)  # You can set your desired seed
        sampled_dataset = shuffled_dataset.select(range(num_examples))
        dataset = sampled_dataset
    elif dataset_name == 'TinyStories':
        # Using the Tiny Stories Data Set
        dataset = load_dataset('roneneldan/TinyStories', split='validation')
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(num_examples))
    elif dataset_name == 'pile_val':
        dataset = load_dataset('json', data_files='data/val.jsonl')
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset['train'].select(range(num_examples))
    elif dataset_name == 'DebugDataSet':
        dataset = load_dataset('json', data_files='data/DebugDataSet.jsonl')
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset['train'].select(range(num_examples))

    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(lambda data: {
        'input_ids': tokenizer.encode(
            data['text'],
            return_tensors='pt'
        ).squeeze(0)
    })
    tokenized_dataset = tokenized_dataset.map(lambda x: {'num_tokens': len(x['input_ids'])})
    chunk_size = prefix_length + suffix_length
    # assert min(tokenized_dataset["num_tokens"]) >= num_buffer + chunk_size
    tokenized_dataset = tokenized_dataset.filter(lambda x: x['num_tokens'] >= num_buffer + chunk_size)
    tokenized_dataset = tokenized_dataset.map(lambda x: {'input_ids_truncated': x['input_ids'][num_buffer:num_buffer+chunk_size]})

    # Get all column names
    all_columns = tokenized_dataset.column_names
    # Find columns to remove
    columns_to_remove = [column for column in all_columns
                            if column != 'input_ids_truncated']
    # Remove unwanted columns
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    tokenized_dataset = tokenized_dataset.rename_column("input_ids_truncated", "input_ids")

    # Use DataCollator to handle padding during training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors='pt'
    )

    # Convert dataset to DataLoader for batch processing
    dataloader = DataLoader(
        tokenized_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size
    )

    return dataloader


def chunk_list(lst, chunk_size):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]



def flatten_dataset(dataset):
    """Flatten a dataset of chunked input ids."""
    flat_dataset = []
    for item in dataset:
        for chunk in item['chunked_input_ids']:
            flat_dataset.append({'input_ids': chunk})
    return flat_dataset


def create_chunked_dataset_from_full_sequences(
        dataset_name,
        tokenizer,
        num_examples,
        prefix_length,
        suffix_length=1,
        batch_size=1,
        seed=42,
        return_all = False
        ):
          
    # Check that the dataset is in the list of valid datasets    
    list_of_dataset_names = ["pile_val", "small-pile-dedup-train", "TinyStories", "DebugDataSet"]
    assert dataset_name in list_of_dataset_names

    chunk_size = prefix_length + suffix_length
    
    if dataset_name == 'small-pile-dedup-train':
        # Using the Pile
        dataset = load_dataset('ola13/small-the_pile-dedup')
        # Concatenate all datasets in the DatasetDict
        concatenated_dataset = concatenate_datasets([ds for ds in dataset.values()])
        # Shuffle the concatenated dataset
        shuffled_dataset = concatenated_dataset.shuffle(seed=42)  # You can set your desired seed
        sampled_dataset = shuffled_dataset.select(range(num_examples))
        dataset = sampled_dataset
    elif dataset_name == 'TinyStories':
        # Using the Tiny Stories Data Set
        dataset = load_dataset('roneneldan/TinyStories', split='validation')
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(num_examples))
    elif dataset_name == 'pile_val':
        dataset = load_dataset('json', data_files='data/val.jsonl')
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset['train'].select(range(num_examples))
    elif dataset_name == 'DebugDataSet':
        dataset = load_dataset('json', data_files='data/DebugDataSet.jsonl')
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset['train'].select(range(num_examples))

    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_and_chunk(data):
        # Tokenize
        input_ids = tokenizer.encode(data['text'], return_tensors='pt').squeeze(0)

        # Chunk tokenized ids
        chunked_input_ids = list(chunk_list(input_ids, chunk_size))

        return {'chunked_input_ids': chunked_input_ids}

    tokenized_dataset = dataset.map(tokenize_and_chunk)
    flat_dataset = flatten_dataset(tokenized_dataset)
    
    if not return_all:
        random.seed(seed)
        flat_dataset = random.sample(flat_dataset, num_examples)


    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors='pt'
    )

    # Convert dataset to DataLoader for batch processing
    dataloader = DataLoader(
        flat_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=batch_size
    )


    return dataloader

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')