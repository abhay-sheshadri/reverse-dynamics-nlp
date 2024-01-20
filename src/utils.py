import torch
from typing import Callable, Iterable, Any
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForCausalLM)


def rand_init(seq_length: int, tokenizer):
    return tokenizer.decode(torch.randint(0, tokenizer.vocab_size, (seq_length,)))


def forward_loss(model, pair, tokenizer, loss=torch.nn.CrossEntropyLoss(),):
    prefix, suffix = pair
    whole_tensor = tokenizer(prefix+suffix, return_tensors='pt').input_ids.cuda()
    with torch.no_grad():
        logs = model(whole_tensor).logits
    start_ind = len(tokenizer.encode(prefix))
    l_pref = loss(logs[0,:start_ind-1], whole_tensor[0,1:start_ind])
    l_suff = loss(logs[0,start_ind-1:-1], whole_tensor[0,start_ind:])
    return l_pref, l_suff


def forward_loss_batch(model, pairs, tokenizer, prefix_len=None, loss=torch.nn.CrossEntropyLoss()):
    if type(pairs) == list:
        prefix_batch, suffix_batch = zip(*pairs)
        whole_text_batch = [prefix + suffix for prefix, suffix in zip(prefix_batch, suffix_batch)]
        whole_tensor = tokenizer(whole_text_batch, return_tensors='pt', padding=True, truncation=True).input_ids.cuda()
    else:
        whole_tensor = pairs.cuda()
    with torch.no_grad():
        logs = model(whole_tensor).logits
    if prefix_len is None:
        start_indices = [len(tokenizer.encode(prefix)) for prefix in prefix_batch]
    else:
        start_indices = [prefix_len] * len(whole_tensor)
    l_pref_batch = []
    l_suff_batch = []
    for (start_ind, whole_tensor_i, logs_i) in zip(start_indices, whole_tensor, logs):
        l_pref = loss(logs_i[:start_ind-1], whole_tensor_i[1:start_ind]) #start_ind=1 case?
        l_suff = loss(logs_i[start_ind-1:-1], whole_tensor_i[start_ind:])
        l_pref_batch.append(l_pref)
        l_suff_batch.append(l_suff)
    return torch.stack(l_pref_batch), torch.stack(l_suff_batch)


def start_chunk_hf(chunk, tokenizer, num_prefix_tokens=10, num_suffix_tokens=40):
    chunk = chunk['text']
    tokens = tokenizer(chunk[:200])['input_ids'] #drop first couple tokens given risk of incomplete token
    yield tokenizer.decode(tokens[:num_prefix_tokens]), tokenizer.decode(tokens[num_prefix_tokens:num_prefix_tokens+num_suffix_tokens])


def get_reverse_pair(dataset: Iterable[Any], chunk_func: Callable[..., Any], tokenizer: AutoTokenizer):
    for chunk in dataset:
        for chunk in chunk_func(chunk, tokenizer):
            yield chunk
