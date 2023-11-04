from datasets import load_dataset, concatenate_datasets
import torch
from einops import rearrange


def create_dataset(
    dataset_name,
    tokenizer,
    num_examples,
    prefix_length,
    suffix_length=1,
    num_buffer=0
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
        dataset = dataset.select(range(num_examples))
    elif dataset_name == 'pile_val':
        dataset = load_dataset('json', data_files='data/val.jsonl')
        dataset = dataset['train'].select(range(num_examples))

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_text(example):
        return {
            'input_ids': tokenizer.encode(
                example['text'],
                truncation=True,
                padding='max_length',
                max_length=sample_length,
                return_tensors='pt'
            ).squeeze(0)
        }

    tokenized_dataset = dataset.map(tokenize_text)

    # Get all column names
    all_columns = tokenized_dataset.column_names
    # Find columns to remove
    columns_to_remove = [column for column in all_columns
                            if column != 'input_ids']
    # Remove unwanted columns
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)


    # Use DataCollator to handle padding during training
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors='pt'
    )

    # Convert dataset to DataLoader for batch processing
    dataloader = DataLoader(tokenized_dataset, shuffle=True,
                            collate_fn=data_collator,
                            batch_size=suffix_batch_size)


# Write stuff for model loading

# Mainly to clean up the other files