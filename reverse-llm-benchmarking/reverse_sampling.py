import math

import torch
import torch.nn.functional as F
from tqdm import tqdm


def get_cond_logprob(input_ids, model):
    # Get conditional logprobs
    with torch.no_grad():
        logprobs = torch.nn.functional.log_softmax(
            model(input_ids=input_ids).logits, dim=-1
        )
    # Get the log probabilities corresponding to the words in input_ids
    relevant_logprobs = torch.gather(
        logprobs, 2, input_ids.unsqueeze(-1)[:, 1:]
    ).squeeze(-1)
    # Sum log probabilities over the sequence length dimension
    sum_log_probs = relevant_logprobs.sum(dim=1)
    return sum_log_probs


def get_logprob(input_ids, model, stationary_dist):
    logprob = torch.log(stationary_dist[input_ids[:, 0]])
    if input_ids.shape[1] > 1:
        logprob = logprob + get_cond_logprob(input_ids, model)
    return logprob


def compute_posterior(
    model,
    stationary_dist,
    tokenized_suffix,
    vocab_batch_size=1024,
    device="cuda"
):

    model.eval()
    vocab_size = stationary_dist.shape[0]
    posterior = []
    total_batches = math.ceil(vocab_size / vocab_batch_size)

    for batch_num in tqdm(range(total_batches)):
        start_idx = batch_num * vocab_batch_size
        end_idx = min(start_idx + vocab_batch_size, vocab_size)

        batch_indices = (
            torch.arange(start_idx, end_idx).to(device)
        )
        v_sentences = torch.cat(
            (batch_indices.unsqueeze(1), tokenized_suffix.repeat(batch_indices.size(0), 1)),
            dim=-1,
        )

        posterior.append(get_logprob(v_sentences, model, stationary_dist))
    
    posterior = torch.cat(posterior)
    return F.log_softmax(posterior, dim=-1)


def sample_with_temp(logits, temperature):
    if temperature == 0:
        p = distribution.argmax()
    else:
        p = torch.distributions.Categorical(
            logits = logits / temperature
        ).sample()
    return p


def sample_reverse_dynamics(
    model,
    stationary_dist,
    prefix_length,
    tokenized_suffix,
    vocab_batch_size=1024,
    temperature=1.0,
    device="cuda"
):
    splus = tokenized_suffix
    full_logits = []
    stationary_dist = stationary_dist.to(device)
    
    for i in range(prefix_length):
        logits = compute_posterior(
            model,
            stationary_dist,
            splus,
            vocab_batch_size,
            device
        )
        full_logits = [logits,] + full_logits
        p = sample_with_temp(
            logits,
            temperature
        )
        splus = torch.cat((p.unsqueeze(0).unsqueeze(0), splus), dim=-1)
        
    return splus, torch.stack(full_logits)


def get_reverse_model_probs(reverse_model, input_ids):
    input_ids = torch.flip(input_ids, (1,))
    outputs = reverse_model(input_ids).logits[0,-1,:]
    return F.softmax(outputs, dim=-1)


def sample_reverse_dynamics_reverse_prior(
    model,
    reverse_model,
    prefix_length,
    tokenized_suffix,
    vocab_batch_size=1024,
    temperature=1.0,
    dilution=0.0,
    device="cuda"
):
    splus = tokenized_suffix
    full_logits = []
    
    for i in range(prefix_length):
        
        # print(tokenizer.decode(splus))

        prior_dist = get_reverse_model_probs(reverse_model, splus)
        
        uniform_dist = torch.ones_like(prior_dist) / prior_dist.shape[0]
        prior_dist = prior_dist * (1-dilution) + uniform_dist * dilution
        
        logits = compute_posterior(
            model,
            prior_dist,
            splus,
            vocab_batch_size,
            device
        )
        full_logits = [logits,] + full_logits
        p = sample_with_temp(
            logits,
            temperature
        )
        splus = torch.cat((p.unsqueeze(0).unsqueeze(0), splus), dim=-1)
        
    return splus, torch.stack(full_logits)


def compute_loss_reverse_dynamics(
    model,
    stationary_dist,
    tokenized_suffix,
    vocab_batch_size=1024,
    dilution=0.0,  # 0.3
    device="cuda",
    loss = F.CrossEntropyLoss()
):
    full_logits = []
    stationary_dist = stationary_dist.to(device)
    
    uniform_dist = torch.ones_like(stationary_dist) / stationary_dist.shape[0]
    stationary_dist = stationary_dist * (1-dilution) + uniform_dist * dilution
    
    for i in reversed(range(1, tokenized_suffix.shape[1])):
        splus = tokenized_suffix[:, i:]        
        
        logits = compute_posterior(
            model,
            stationary_dist,
            splus,
            vocab_batch_size,
            device
        )
        full_logits = [logits,] + full_logits
            
    logits = torch.stack(full_logits).to(tokenized_suffix.device)
    return loss(logits, tokenized_suffix[0, :-1]).item()


def compute_loss_reverse_dynamics_reverse_prior(
    model,
    reverse_model,
    tokenized_suffix,
    vocab_batch_size=1024,
    dilution=0.0,  # 0.3
    device="cuda",
    loss = F.CrossEntropyLoss()
):
    full_logits = []
    
    for i in reversed(range(1, tokenized_suffix.shape[1])):
        splus = tokenized_suffix[:, i:]

        prior_dist = get_reverse_model_probs(reverse_model, splus)
        
        uniform_dist = torch.ones_like(prior_dist) / prior_dist.shape[0]
        prior_dist = prior_dist * (1-dilution) + uniform_dist * dilution
        
        logits = compute_posterior(
            model,
            prior_dist,
            splus,
            vocab_batch_size,
            device
        )
        full_logits = [logits,] + full_logits
            
    logits = torch.stack(full_logits).to(tokenized_suffix.device)
    
    return loss(logits, tokenized_suffix[0, :-1]).item()
