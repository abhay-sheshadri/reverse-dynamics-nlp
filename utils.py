import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForCausalLM)
from transformers.generation.logits_process import (LogitsProcessor,
                                                    LogitsProcessorList)
from datasets import load_dataset
from typing import Callable, Iterable, Any
import matplotlib.pyplot as plt

SOFTMAX_FINAL = nn.Softmax(dim=-1)
LOGSOFTMAX_FINAL = nn.LogSoftmax(dim=-1)
CROSSENT = nn.CrossEntropyLoss(reduction='none')



def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    embed_weights = list(model.modules())[2]
    assert type(embed_weights).__name__=='Embedding'
    embed_weights = embed_weights.weight
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = model.get_input_embeddings()(input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:],
            input_embeds,
            embeds[:,input_slice.stop:,:]
        ],
        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

    loss.backward()

    return one_hot.grad.clone()


def token_gradients_with_output(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.
    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    embed_weights = list(model.modules())[2]
    assert type(embed_weights).__name__=='Embedding'
    embed_weights = embed_weights.weight
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # now stitch it together with the rest of the embeddings
    embeds = model.get_input_embeddings()(input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:],
            input_embeds,
            embeds[:,input_slice.stop:,:]
        ],
        dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

    loss.backward()

    return one_hot.grad.clone(), logits.detach().clone()


def reverse_tokenize(tokenizer, target):
    input_ids = tokenizer.encode(target, return_tensors="pt").cuda()
    input_ids = torch.flip(input_ids, (1,))
    return input_ids


def reverse_output(output):
    return torch.flip(output, (1,))


def reverse_decode(tokenizer, output):
    tokens = torch.flip(output, (1,))
    return [
        tokenizer.decode(tokens[i]) for i in range(tokens.shape[0])
    ]


def reverse_normalized_forward(reverse_model, tokenizer, target, normalizer=None):
    inputs = reverse_tokenize(tokenizer, target)
    outputs = reverse_model(inputs).logits[0,-1,:]
    outputs = SOFTMAX_FINAL(outputs).cpu()
    if not normalizer is None:
        outputs = torch.mul(outputs, normalizer)
    return outputs


def reverse_normalized_generate(reverse_model, tokenizer, target, max_length, normalizer=None, temperature=1):
    prefix = []
    for i in range(max_length):
        normalized_probs = reverse_normalized_forward(reverse_model, tokenizer, ''.join(prefix[::-1]) + target, normalizer)
        if not temperature:
            token = tokenizer.decode(torch.argmax(normalized_probs))
        else:
            probs = torch.div(normalized_probs, temperature)
            probs = torch.nn.Softmax(dim=-1)(probs)
            token = tokenizer.decode(torch.multinomial(probs, num_samples=1))
        if token == '[PAD]' or token == '[EOS]':
            break
        prefix.append(token)
    return ''.join(prefix[::-1])+target


def reverse_positional_generate(reverse_model, tokenizer, target, max_length, normalizer=None, temperature=1):
    prefix = []
    for i in range(max_length):
        normalized_probs = reverse_positional_forward(reverse_model, tokenizer, ''.join(prefix[::-1]) + target, max_length-i-1, normalizer)
        if not temperature:
            token = tokenizer.decode(torch.argmax(normalized_probs))
        else:
            probs = torch.div(normalized_probs, temperature)
            probs = torch.nn.Softmax(dim=-1)(probs)
            token = tokenizer.decode(torch.multinomial(probs, num_samples=1))
        if token == '[PAD]' or token == '[EOS]':
            break
        prefix.append(token)
    return ''.join(prefix[::-1])+target


class SampleTopTokens(LogitsProcessor):

    def __init__(self, n_initial_tokens, n_new_tokens, top_grad_tokens):
        self.n_initial_tokens = n_initial_tokens
        self.n_new_tokens = n_new_tokens
        self.top_grad_tokens = top_grad_tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        curr_pos = self.n_new_tokens - (input_ids.shape[-1] - self.n_initial_tokens) - 1
        mask = torch.ones(scores.shape, dtype=torch.bool, device=scores.device)
        mask[:, self.top_grad_tokens[curr_pos]] = False
        scores.masked_fill_(mask, -float('inf'))
        return scores


def get_pos_token_probabilities(tokenizer, dataset="NeelNanda/pile-10k", vocab_size=50304, split='train', prefix=10, prev_counts=None):
    if type(dataset) == str:
        data = load_dataset(dataset)
    else:
        data = dataset
    if prev_counts is None:
        counts = torch.zeros((vocab_size, prefix), dtype=torch.float)
    else: 
        counts = prev_counts
    
    token_to_string_rough_bound = 10*prefix
    for chunk in data[split]:
        text = chunk['text']
        tokens = tokenizer(text[:token_to_string_rough_bound], return_tensors="pt").input_ids[0]
        tokens = tokens[:prefix]
        if len(tokens) < prefix:
            tokens = tokenizer(text, return_tensors="pt").input_ids[0][:prefix]
        for t,token in enumerate(tokens):
            counts[token,t] += 1
    return counts


def start_chunk_hf(chunk, tokenizer, num_prefix_tokens=10, num_suffix_tokens=25):
    chunk = chunk['text']
    tokens = tokenizer(chunk[:200])['input_ids'] #drop first couple tokens given risk of incomplete token
    yield tokenizer.decode(tokens[:num_prefix_tokens]), tokenizer.decode(tokens[num_prefix_tokens:num_prefix_tokens+num_suffix_tokens])


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


def get_reverse_pair(dataset: Iterable[Any], chunk_func: Callable[..., Any], tokenizer: AutoTokenizer):
    for chunk in dataset:
        for chunk in chunk_func(chunk, tokenizer):
            yield chunk


def get_pos_token_probabilities_pandas(tokenizer, dataset, vocab_size=50304, prefix=10, prev_counts=None):
    if prev_counts is None:
        counts = torch.zeros((vocab_size, prefix), dtype=torch.float)
    else: 
        counts = prev_counts
    
    token_to_string_rough_bound = 10*prefix
    for chunk in dataset:
        text = chunk[1] #1 is assumed to be first data column having text
        tokens = tokenizer(text[:token_to_string_rough_bound], return_tensors="pt").input_ids[0]
        tokens = tokens[:prefix]
        if len(tokens) < prefix:
            tokens = tokenizer(text, return_tensors="pt").input_ids[0][:prefix]
            if len(tokens) < prefix:
                continue
        for t,token in enumerate(tokens):
            counts[token,t] += 1
    return counts


def get_token_probabilities_pandas(tokenizer, dataset="NeelNanda/pile-10k", vocab_size=50304, split='train', prev_counts=None):
    # if type(dataset)==str:
    #     data = load_dataset(dataset)
    # else:
    data = dataset
    if prev_counts is None:
        counts = torch.zeros(vocab_size, dtype=torch.float) #tokenizer.vocab_size is fake 50304 is the model output dimension which is what we care about
    else:
        counts = prev_counts
    for chunk in data:
        text = chunk[1]
        tokens = tokenizer(text, return_tensors="pt").input_ids[0]
        token_counts = torch.bincount(tokens.type(torch.long), minlength=counts.size(0))
        counts += token_counts

    # total_tokens = torch.sum(counts)
    # probabilities = counts / total_tokens
    # min_val = probabilities[probabilities > 0].min()
    # probabilities[probabilities == 0] = min_val
    return counts


def reverse_tokenize_batch(tokenizer, targets):
    input_ids = tokenizer(targets, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    input_ids = torch.flip(input_ids, (1,))
    return input_ids


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

def reverse_positional_forward(reverse_model, tokenizer, targets, pos, normalizer=None):
    if type(targets) == list:
        inputs = reverse_tokenize_batch(tokenizer, targets)  # Assume this function can handle batched targets
    else:
        inputs = torch.flip(targets, (1,)).cuda()
    with torch.no_grad():
        outputs = reverse_model(inputs).logits[:, -1, :]  # Adjust indexing for batched outputs
    outputs = SOFTMAX_FINAL(outputs).cpu()
    if normalizer is not None:
        outputs = torch.mul(outputs, normalizer[:, pos])
    return outputs


def reverse_normalized_beam_generate(reverse_model, tokenizer, target, max_length, beam_size=10, normalizer=None):
    beams = [(1, torch.empty(0,))]  # List of tuples of score and prefix
    target_tokens = tokenizer(target, return_tensors="pt",).input_ids[0]
    for i in range(max_length):
        targets = torch.stack([torch.cat((prefix,target_tokens)) for _, prefix in beams]).type(target_tokens.dtype).cuda()
        normalized_probs = reverse_positional_forward(reverse_model, tokenizer, targets, max_length-i-1, normalizer)
        candidates = []
        for j, (score, prefix) in enumerate(beams):
            probs, indices = torch.topk(normalized_probs[j], beam_size)  # Get top-k probabilities and indices for each beam
            for prob, idx in zip(probs, indices):  # No batch dimension here, handled in the outer loop
                new_prefix = torch.cat((idx.view(1,),prefix))
                new_score = score*prob.item()
                candidates.append((new_score, new_prefix))
        # Sort candidates by score and keep the top-k
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]
    return [b[1].type(target_tokens.dtype) for b in beams]


def reverse_fwd_beam_generate(reverse_model, forward_model, tokenizer, target, max_length, beam_size=10, normalizer=None):
    beams = [(1, torch.empty(0, dtype=torch.long))]  # List of tuples of score and prefix
    target_tokens = tokenizer(target, return_tensors="pt").input_ids[0]  
    for i in range(max_length):
        targets = torch.stack([torch.cat((prefix, target_tokens)) for _, prefix in beams]).type(target_tokens.dtype)
        normalized_probs = reverse_positional_forward(reverse_model, tokenizer, targets, max_length-i-1, normalizer).cpu()
        candidates = []
        for j, (_, prefix) in enumerate(beams):
            _, indices = torch.topk(normalized_probs[j], beam_size)  # Get top-k probabilities and indices for each beam
            for idx in indices:
                new_prefix = torch.cat((idx.view(1,), prefix))
                candidates.append(new_prefix)
        
        pairs_batch = torch.stack(candidates)
        pairs_batch = torch.cat((pairs_batch, target_tokens.repeat(pairs_batch.shape[0],1)), dim=1)
        
        _, l_suff = forward_loss_batch(forward_model, pairs_batch, tokenizer, prefix_len=i+1)  # Assuming all prefixes have the same length
        
        # Update candidates with new scores based on forward model loss
        candidates = [(l_suff[i].item(), candidates[i]) for i in range(len(candidates))]
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]
    
    return [b[1].type(target_tokens.dtype) for b in beams]  # Returning tensors as in reverse_normalized_beam_generate

def plot_beams(all_losses, all_naturals, beam_size, normalizer_temp, base_prefix_loss=None, base_suffix_loss=None):
    eval_size = len(all_losses)
    print(f'inverse dataset probs temp is {normalizer_temp}')

    prefix_loss_at_n, best_suffix_loss_at_n = [[loss[0]] for loss in all_naturals], [[loss[0]] for loss in all_losses]

    # For each beam check iterate over all samples and check whether the loss on that beam+sample improved over previous best on that sample.
    for n in range(beam_size):
        if n == 0:
            continue
        for l,loss_list in enumerate(all_losses):
            next_suffix_loss = loss_list[n]
            if next_suffix_loss < best_suffix_loss_at_n[l][-1]:
                best_suffix_loss_at_n[l].append(next_suffix_loss)
                prefix_loss_at_n[l].append(all_naturals[l][n])
            else:
                best_suffix_loss_at_n[l].append(best_suffix_loss_at_n[l][-1])
                prefix_loss_at_n[l].append(prefix_loss_at_n[l][-1])
    
    suffix_loss_mat = np.array(best_suffix_loss_at_n)
    prefix_loss_mat = np.array(prefix_loss_at_n)
    assert suffix_loss_mat.shape[0] == eval_size
    mean_prefix_losses = np.mean(prefix_loss_mat, axis=0)
    mean_suffix_losses = np.mean(suffix_loss_mat, axis=0)

# Plotting
    plt.figure()
    plt.plot(mean_prefix_losses, mean_suffix_losses, marker='o', label='Best-of-N')
    plt.plot([mean_prefix_losses[0]], [mean_suffix_losses[0]], marker='x', linestyle='', color='red', label='Greedy Prefix')
    if base_prefix_loss is not None:
        plt.plot([base_prefix_loss], [base_suffix_loss], marker='s', linestyle='', color='green', label='Dataset Prefix')
    plt.xlabel(f'Forwards LM Prefix Loss')
    plt.ylabel('Forwards LM Suffix Loss')
    plt.title(f'Num beams varies from 1 to {beam_size}, mean over {eval_size} samples')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()