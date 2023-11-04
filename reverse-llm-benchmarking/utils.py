from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import (DataCollatorForLanguageModeling, GPTNeoXForCausalLM,
                          GPTNeoXTokenizerFast)


def create_dataset(
    dataset_name,
    tokenizer,
    num_examples,
    prefix_length,
    suffix_length=1,
    num_buffer=0,
    suffix_batch_size=1,
    seed=42
):
          
    # Check that the dataset is in the list of valid datasets
    list_of_dataset_names = ["pile_val", "small-pile-dedup-train", "TinyStories"]
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

    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = dataset.map(lambda data: {
        'input_ids': tokenizer.encode(
            data['text'],
            return_tensors='pt'
        ).squeeze(0)
    })
    tokenized_dataset = tokenized_dataset.map(lambda x: {'num_tokens': len(x['input_ids'])})
    total_length = prefix_length + suffix_length - 1
    assert min(tokenized_dataset["num_tokens"]) >= num_buffer + total_length
    tokenized_dataset = tokenized_dataset.map(lambda x: {'input_ids_truncated': x['input_ids'][num_buffer:num_buffer+total_length]})

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
        batch_size=suffix_batch_size
    )

    return dataloader


# Write stuff for model loading

# Mainly to clean up the other files