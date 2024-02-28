import torch
from transformers import AutoTokenizer
from utils import get_token_probabilities_pandas
import pandas as pd
from joblib import Parallel, delayed

tokenizer = AutoTokenizer.from_pretrained("afterless/reverse-pythia-160m")

def process_file(i, end=10):
    prev_counts = None
    num = str(i).zfill(2)
    print(num)
    file_path = f'/vast/work/public/ml-datasets/pile/train/{num}.jsonl'
    json_reader = pd.read_json(file_path, lines=True, chunksize=100000)  # Adjust chunksize as needed
    
    for c,chunk in enumerate(json_reader):
        print(c)
        prev_counts = get_token_probabilities_pandas(tokenizer, dataset=chunk.itertuples(), prev_counts=prev_counts)
        if c == end:
            break
    
    with open(f'/home/jp6263/reverse-dynamics-nlp/abs_counts_{num}.pt', 'wb') as f:
        torch.save(prev_counts, f)

Parallel(n_jobs=-1)(delayed(process_file)(i) for i in range(0,10))
