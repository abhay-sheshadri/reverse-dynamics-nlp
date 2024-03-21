import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForCausalLM)
from src.utils import *


class ReversalLMPrior:

    def __init__(
        self,
        model: AutoModelForCausalLM,
        reverse_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size=1024,
        num_top_tokens: int = 10_000,
    ):

        self.model = model
        self.reverse_model = reverse_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_top_tokens = num_top_tokens

    def sample_proposals(
        self,
        input_length,
        target_ids,
        temperature = None
    ):
        # Sample random proposals
        if temperature is None:
            temperature = 1.0
        proposals = []
        tokens, _ = sample_reverse_dynamics_reverse_prior(
            self.model,
            self.reverse_model,
            prefix_length=input_length,
            tokenized_suffix=target_ids,
            vocab_batch_size=self.batch_size,
            temperature=temperature,
            dilution=0.3,
            device="cuda",
            num_top_tokens=self.num_top_tokens
        )
        return tokens

    def optimize(
        self,
        initial_input,
        target_string,
        use_prefix_loss=True,
        temperature=0,
    ):
        # Parse input strings into tokens
        initial_inputs = self.tokenizer.encode(initial_input, return_tensors="pt").cuda()
        initial_targets = self.tokenizer.encode(target_string, return_tensors="pt").cuda()
        # Sample proposals
        proposals = self.sample_proposals(initial_inputs.shape[-1], initial_targets, temperature=temperature)
        # Choose the proposal with the lowest loss
        return self.tokenizer.decode(proposals[0])
    

class ReversalEmpiricalPrior:

    def __init__(
        self,
        model: AutoModelForCausalLM,
        dist: torch.Tensor,
        tokenizer: AutoTokenizer,
        batch_size=1024,
        reverse_model: AutoModelForCausalLM = None,
        num_top_tokens: int = 10_000,

    ):

        self.model = model
        self.dist = dist
        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.reverse_model = reverse_model
        self.num_top_tokens = num_top_tokens

    def sample_proposals(
        self,
        input_length,
        target_ids,
        temperature = None
    ):
        # Sample random proposals
        if temperature is None:
            temperature = 1.0
        proposals = []
        tokens, _ = sample_reverse_dynamics(
            self.model,
            self.dist,
            prefix_length=input_length,
            tokenized_suffix=target_ids,
            vocab_batch_size=self.batch_size,
            temperature=temperature,
            dilution=0.3,
            device="cuda",
            reverse_model=self.reverse_model,
            num_top_tokens=self.num_top_tokens
        )
        return tokens

    def optimize(
        self,
        initial_input,
        target_string,
        temperature=0.7,
    ):
        # Parse input strings into tokens
        initial_inputs = self.tokenizer.encode(initial_input, return_tensors="pt").cuda()
        initial_targets = self.tokenizer.encode(target_string, return_tensors="pt").cuda()
        # Sample proposals
        proposals = self.sample_proposals(initial_inputs.shape[-1], initial_targets, temperature=temperature)
        # Choose the proposal with the lowest loss
        return self.tokenizer.decode(proposals[0])


def get_cond_logprob(input_ids, model):
    # Get conditional logprobs
    with torch.no_grad():
        logprobs = torch.nn.functional.log_softmax(
            model(input_ids=input_ids[:,:-1]).logits, dim=-1
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
    device="cuda",
    indices=None,
    disable_tqdm=True
):
    model.eval()
    vocab_size = stationary_dist.shape[0]
    posterior = []
    
    if indices is None:
        full_indices = torch.arange(0, vocab_size, device=device)
    else:
        full_indices = indices.to(device)

    total_batches = math.ceil(full_indices.shape[-1] / vocab_batch_size)

    for batch_num in tqdm(range(total_batches),disable=disable_tqdm):
        start_idx = batch_num * vocab_batch_size
        end_idx = min(start_idx + vocab_batch_size, vocab_size)

        batch_indices = full_indices[start_idx:end_idx]
        v_sentences = torch.cat(
            (batch_indices.unsqueeze(1), tokenized_suffix.repeat(batch_indices.size(0), 1)),
            dim=-1,
        )

        posterior.append(get_logprob(v_sentences, model, stationary_dist))
    
    posterior = torch.cat(posterior)
    posterior[torch.isnan(posterior)] = -100
    posterior = F.log_softmax(posterior, dim=-1)
    
    if indices is not None:
        new_post = torch.ones_like(stationary_dist) * -100000
        new_post[indices] = posterior
        return new_post
    else:
        return posterior


def sample_with_temp(logits, temperature):
    if temperature == 0:
        p = logits.argmax()
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
    dilution=0.0,
    device="cuda",
    reverse_model=None,
    num_top_tokens=10_000,
    disable_tqdm=True
):
    splus = tokenized_suffix
    full_logits = []
    prior_dist = stationary_dist.to(device)
    
    uniform_dist = torch.ones_like(prior_dist) / prior_dist.shape[0]
    prior_dist = prior_dist * (1-dilution) + uniform_dist * dilution
    
    for i in range(prefix_length):
        
        if reverse_model is not None:
            _, possible_tokens = get_reverse_model_probs(reverse_model, splus, num_top_tokens)
        else:
            possible_tokens = None
        
        logits = compute_posterior(
            model=model,
            stationary_dist=prior_dist,
            tokenized_suffix=splus,
            vocab_batch_size=vocab_batch_size,
            device=device,
            disable_tqdm=disable_tqdm
        )

        full_logits = [logits,] + full_logits
        p = sample_with_temp(
            logits,
            temperature
        )
        splus = torch.cat((p.unsqueeze(0).unsqueeze(0), splus), dim=-1)
        
    return splus, torch.stack(full_logits)


def get_reverse_model_probs(reverse_model, input_ids, num_top_tokens=None, filter_prob=None):
    input_ids = torch.flip(input_ids, (1,))
    outputs = reverse_model(input_ids).logits[0,-1,:]
    probs = F.softmax(outputs, dim=-1)
    if filter_prob is not None:
        filter = (probs > filter_prob)
        probs = probs * filter
        top_tokens = torch.nonzero(filter, as_tuple=True)[0]
        return probs, top_tokens
    elif num_top_tokens is not None:
        top_tokens = outputs.sort(descending=True).indices[:num_top_tokens]
        return probs, top_tokens
    else:
        return probs, None


def sample_reverse_dynamics_reverse_prior(
    model,
    reverse_model,
    prefix_length,
    tokenized_suffix,
    vocab_batch_size=1024,
    temperature=1.0,
    dilution=0.0,
    device="cuda",
    num_top_tokens=None,
    filter_prob=None,
    disable_tqdm=True
):
    splus = tokenized_suffix
    full_logits = []
    
    for i in range(prefix_length):
        
        # print(tokenizer.decode(splus))

        prior_dist, possible_tokens = get_reverse_model_probs(reverse_model, splus, num_top_tokens=num_top_tokens, filter_prob=filter_prob)
        
        uniform_dist = torch.ones_like(prior_dist) / prior_dist.shape[0]
        prior_dist = prior_dist * (1-dilution) + uniform_dist * dilution
        
        logits = compute_posterior(
            model=model,
            stationary_dist=prior_dist,
            tokenized_suffix=splus,
            vocab_batch_size=vocab_batch_size,
            device=device,
            indices=possible_tokens,
            disable_tqdm=disable_tqdm
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
    loss = torch.nn.CrossEntropyLoss()
):
    full_logits = []
    stationary_dist = stationary_dist.to(device)
    suffix_length = tokenized_suffix.shape[1]
    if len(stationary_dist.shape) == 1:
        multiple_priors = False
    elif len(stationary_dist.shape) == 2:
        assert stationary_dist.shape[1] == suffix_length-1
        multiple_priors = True
    else:
        raise Exception("Tensor of priors is not the correct shape.")
    
    uniform_dist = torch.ones_like(stationary_dist) / stationary_dist.shape[0]
    stationary_dist = stationary_dist * (1-dilution) + uniform_dist * dilution
    
    for i in reversed(range(1, tokenized_suffix.shape[1])):
        splus = tokenized_suffix[:, i:] 

        if multiple_priors:   
            current_prior = stationary_dist[:, i-1]
        else:   
            current_prior = stationary_dist
        
        logits = compute_posterior(
            model,
            current_prior,
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
    loss = torch.nn.CrossEntropyLoss(),
    disable_tqdm = False
):
    full_logits = []
    
    for i in tqdm(reversed(range(1, tokenized_suffix.shape[1])),disable=disable_tqdm):
        splus = tokenized_suffix[:, i:]

        prior_dist, _ = get_reverse_model_probs(reverse_model, splus)
        
        uniform_dist = torch.ones_like(prior_dist) / prior_dist.shape[0]
        prior_dist = prior_dist * (1-dilution) + uniform_dist * dilution
        
        logits = compute_posterior(
            model,
            prior_dist,
            splus,
            vocab_batch_size,
            device,
            disable=disable_tqdm
        )
        full_logits = [logits,] + full_logits
            
    logits = torch.stack(full_logits).to(tokenized_suffix.device)
    
    return loss(logits, tokenized_suffix[0, :-1]).item()


def compute_loss_reverse_dynamics_reverse_prior_target_memory(
    model,
    reverse_model,
    tokenized_suffix,
    target_memory = 10.0, # in gigabytes 
    dilution=0.0,  # 0.3
    device="cuda",
    loss = torch.nn.CrossEntropyLoss(),
    disable_tqdm=True
):
    full_logits = []
    
    for i in tqdm(reversed(range(1, tokenized_suffix.shape[1])), disable=disable_tqdm):
        splus = tokenized_suffix[:, i:]

        prior_dist = get_reverse_model_probs(reverse_model, splus)
        
        uniform_dist = torch.ones_like(prior_dist) / prior_dist.shape[0]
        prior_dist = prior_dist * (1-dilution) + uniform_dist * dilution
        
        logits = compute_posterior(
            model,
            prior_dist,
            splus,
            math.ceil(target_memory * 1e9/(4*(tokenized_suffix.shape[1]-i)*(50304))),
            device,
            disable=disable_tqdm
        )
        full_logits = [logits,] + full_logits
            
    logits = torch.stack(full_logits).to(tokenized_suffix.device)
    
    return loss(logits, tokenized_suffix[0, :-1]).item()