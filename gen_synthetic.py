import pickle
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("afterless/reverse-pythia-160m")
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m", cache_dir='/scratch/jp6263/hf/models/').cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset("NeelNanda/pile-10k")
data = dataset["train"]
PREFIX_LENGTH = 4
EXAMPLES = 2000
REPEAT = int(1e6) #1e6 in paper, 3e5 is ~1min per
EXTRA_TOKENS = 2 # or 5
BATCH=int(1e4)
FREQUENCY_THRESHOLD = 0.9
# Set the device to GPU if available, else CPU

# Initialize an empty dictionary to store the test sets
testSet = {}

# Create a randomized list of indices up to the size of the data, 
# then take the first 'EXAMPLES' number of indices and convert to a list
indices = torch.randperm(len(data))[:EXAMPLES].tolist()

for i in tqdm(indices):
    if len(testSet)==200: break
    # Encode the text data of the current item, limit to PREFIX_LENGTH, and adjust tensor dimensions

    textData = tokenizer.encode(data[i]["text"], return_tensors="pt")[:, EXTRA_TOKENS:PREFIX_LENGTH+EXTRA_TOKENS]  # get random prefixes indexed by random indices i
    if len(textData[0]) < PREFIX_LENGTH:
        continue
    print(f'prefix: {tokenizer.decode(textData[0])}')
    textData.to(device)

    # Initialize empty tensors with specified dimensions on the GPU for storing combined sequences and generated tokens
    globalKey = torch.empty((0, EXTRA_TOKENS + PREFIX_LENGTH), dtype=torch.long).to(device)
    out = torch.empty((0,), dtype=torch.long).to(device)

    low_entropy = True
    for ind in range(0, REPEAT, BATCH):
        # Create random tokens, concatenate with repeated prefix data, and move to GPU
        key = torch.cat([torch.randint(0, tokenizer.vocab_size, (BATCH, EXTRA_TOKENS),), textData.repeat(BATCH, 1)], dim=-1).to(device)
        # Generate new tokens using the model, based on the combined random and prefix data, then concatenate to 'out' tensor
        out = torch.cat([out, model.generate(key, do_sample=False, num_beams=1, max_length=key.shape[1]+1)[:, -1]])  # (REPEAT, 1)
        # Concatenate the combined sequences to 'globalKey' tensor
        globalKey = torch.cat([globalKey, key], dim=0)
        if ind==0:
            freqs = torch.bincount(out, minlength=tokenizer.vocab_size)
            if max(freqs)<BATCH*FREQUENCY_THRESHOLD: 
                print(max(freqs)/BATCH, 'skipping')
                low_entropy=False
                break
            else:
                print(f'Acceptable: prefix has max prob on {tokenizer.decode(torch.argmax(freqs))}')
    if not low_entropy: continue
    
    freqs = torch.bincount(out, minlength=tokenizer.vocab_size)
    probs = freqs / freqs.sum() 
    lowProbs = ((0 < probs) & (probs <= 1/REPEAT)).nonzero(as_tuple=True)[0]  # 1e-7

    # Concatenate 'globalKey' and 'out' tensors, adjust dimensions of 'out' to match
    tmp = torch.cat([globalKey, out.unsqueeze(1)], dim=-1)
    # Identify the indices of rows in 'tmp' where the last token is a low probability token
    lowProbIndices = torch.isin(tmp[:, -1], lowProbs)
    # Use these indices to select the corresponding rows from 'tmp'
    selectedRows = tmp[lowProbIndices]
    testSet[i] = selectedRows.cpu()

with open("/home/jp6263/reverse-dynamics-nlp/entropy_testSet.pkl", "wb") as f:
    pickle.dump(testSet, f)