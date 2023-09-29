import pickle
import torch as t
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
dataset = load_dataset("NeelNanda/pile-10k")
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m",).to(device)

data = dataset["train"]
PREFIX_LENGTH = 5
EXAMPLES = 200
REPEAT = int(1e4)
EXTRA_TOKENS = 2
BATCH=int(1e3)
# Set the device to GPU if available, else CPU

# Initialize an empty dictionary to store the test sets
testSet = {}

# Create a randomized list of indices up to the size of the data, 
# then take the first 'EXAMPLES' number of indices and convert to a list
indices = t.randperm(len(data))[:EXAMPLES].tolist()

for i in tqdm(indices):
    # Encode the text data of the current item, limit to PREFIX_LENGTH, and adjust tensor dimensions
    print(f'prefix is: {data[i]["text"][:PREFIX_LENGTH*8]}')
    textData = tokenizer.encode(data[i]["text"], return_tensors="pt")[:, :PREFIX_LENGTH]  # get random prefixes indexed by random indices i
    if len(textData[0]) < PREFIX_LENGTH:
        continue
    textData.to(device)

    # Initialize empty tensors with specified dimensions on the GPU for storing combined sequences and generated tokens
    globalKey = t.empty((0, EXTRA_TOKENS + PREFIX_LENGTH), dtype=t.long).to(device)
    out = t.empty((0,), dtype=t.long).to(device)

    for _ in range(0, REPEAT, BATCH):
        # Create random tokens, concatenate with repeated prefix data, and move to GPU
        key = t.cat([t.randint(0, tokenizer.vocab_size, (BATCH, EXTRA_TOKENS),), textData.repeat(BATCH, 1)], dim=-1).to(device)
        # Generate new tokens using the model, based on the combined random and prefix data, then concatenate to 'out' tensor
        out = t.cat([out, model.generate(key, do_sample=False, num_beams=1, max_length=key.shape[1]+1)[:, -1]])  # (REPEAT, 1)
        # Concatenate the combined sequences to 'globalKey' tensor
        globalKey = t.cat([globalKey, key], dim=0)
    freqs = t.bincount(out, minlength=tokenizer.vocab_size)
    probs = freqs / freqs.sum() 
    lowProbs = ((0 < probs) & (probs <= 1/REPEAT)).nonzero(as_tuple=True)[0]  # 1e-7

    # Concatenate 'globalKey' and 'out' tensors, adjust dimensions of 'out' to match
    tmp = t.cat([globalKey, out.unsqueeze(1)], dim=-1)
    # Identify the indices of rows in 'tmp' where the last token is a low probability token
    lowProbIndices = t.isin(tmp[:, -1], lowProbs)
    # Use these indices to select the corresponding rows from 'tmp'
    selectedRows = tmp[lowProbIndices]
    testSet[i] = selectedRows.cpu()

with open("/home/jp6263/reverse-dynamics-nlp/short_testSet.pkl", "wb") as f:
    pickle.dump(testSet, f)