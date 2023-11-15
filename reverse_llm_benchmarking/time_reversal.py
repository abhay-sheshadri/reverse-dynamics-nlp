import numpy as np
from tqdm import tqdm
import time
import math
import gc
import torch
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, DataCollatorForLanguageModeling
from reverse_sampling import sample_reverse_dynamics_reverse_prior
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-12B-deduped",
).to(device)
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

reverse_model = GPTNeoXForCausalLM.from_pretrained(
    "afterless/reverse-pythia-160m"
).to(device)

# current time
tokenized_suffix = tokenizer.encode(" basketball")
print(tokenized_suffix.shape)
# current time
start = time.time()
out = sample_reverse_dynamics_reverse_prior(
    model,
    reverse_model,
    10,
    tokenized_suffix,
    vocab_batch_size=3000,
    temperature=1.0,
    dilution=0.0,
    device="cuda"
)
print(out)
end = time.time()
print(end - start)






