import gc
import random
import pickle
from collections import defaultdict

import numpy as np
import torch as t
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset

dataset = load_dataset("NeelNanda/pile-10k")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m", cache_dir=".cache/models")

data = dataset["train"]

PREFIX_LENGTH = 10
EXAMPLES = 150
REPEAT = 1_000_000
EXTRA_TOKENS = 5

testSet = {}
indices = t.randperm(len(data))[:EXAMPLES].tolist()
for i in indices:
    textData = tokenizer.encode(data[i]["text"], return_tensors="pt")[:, :PREFIX_LENGTH]
    globalKey = t.empty((0, EXTRA_TOKENS + PREFIX_LENGTH), dtype=t.long)
    out = t.empty((0,), dtype=t.long)
    for j in range(0, REPEAT, 25000):
        key = t.cat([t.randint(0, tokenizer.vocab_size, (25000, EXTRA_TOKENS)), textData.repeat(25000, 1)], dim=-1)
        out = t.cat([out, model.generate(key, do_sample=False, num_beams=1, max_length=key.shape[1]+1)[:, -1]]) # (REPEAT, 1)
        globalKey = t.cat([globalKey, key], dim=0)
        del key
        gc.collect()


    freqs = t.bincount(out, minlength=tokenizer.vocab_size)
    probs = freqs / freqs.sum()
    lowProbs = ((0 < probs) & (probs <= 10e-6)).nonzero()
    tmp = t.cat([globalKey, out.unsqueeze(1)], dim=-1)
    testSet[i] = t.empty((0, tmp.shape[1]), dtype=t.long)
    for r in tmp:
        if r[-1] in lowProbs:
            testSet[i] = t.cat([testSet[i], r.unsqueeze(0)], dim=0)
    print(testSet)

with open("testSet.pkl", "wb") as f:
    pickle.dump(testSet, f)

del testSet