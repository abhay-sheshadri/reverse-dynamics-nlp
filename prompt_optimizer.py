import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForCausalLM)
from utils import *


class PromptOptimizer:
    """
    Implementation of Default GCG method
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        n_proposals: int = 128,
        n_epochs: int = 256,
        n_top_indices: int = 128
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.n_proposals = n_proposals
        self.n_epochs = n_epochs
        self.n_top_indices = n_top_indices

    def calculate_restricted_subset(
        self,
        input_ids,
        input_slice,
        target_slice,
        loss_slice
    ):
        # Find the subset of tokens that have the most impact on the likelihood of the target
        grad = token_gradients(self.model, input_ids, input_slice, target_slice, loss_slice)
        top_indices = torch.topk(-grad, self.n_top_indices, dim=-1).indices
        top_indices = top_indices.detach().cpu().numpy()
        return top_indices

    def sample_proposals(
        self,
        input_ids,
        top_indices,
        input_slice,
        target_slice,
        loss_slice
    ):
        # Sample random proposals
        proposals = []
        for p in range(self.n_proposals):
            rand_idx = np.random.randint(input_slice.start, input_slice.stop)
            rand_token = np.random.choice(top_indices[rand_idx])
            prop = input_ids.clone()
            prop[rand_idx] = rand_token
            proposals.append(prop)
        return proposals

    def optimize(
        self,
        initial_input,
        target_string
    ):
        # Parse input strings into tokens
        initial_inputs = self.tokenizer.encode(initial_input, return_tensors="pt")[0].cuda()
        initial_targets = self.tokenizer.encode(target_string, return_tensors="pt")[0].cuda()
        input_ids = torch.cat([initial_inputs, initial_targets], dim=0)
        input_slice = slice(0, initial_inputs.shape[0])
        target_slice = slice(initial_inputs.shape[0], input_ids.shape[-1])
        loss_slice = slice(initial_inputs.shape[0] - 1, input_ids.shape[-1] - 1)
        # Optimize input
        prev_loss = None
        for i in range(self.n_epochs):
            # Get proposals for next string
            top_indices = self.calculate_restricted_subset(input_ids, input_slice, target_slice, loss_slice)
            proposals = self.sample_proposals(input_ids, top_indices, input_slice, target_slice, loss_slice)
            # Choose the proposal with the lowest loss
            with torch.no_grad():
                proposals = torch.stack(proposals)
                prop_logits = self.model(proposals).logits
                targets = input_ids[target_slice]
                losses = [nn.CrossEntropyLoss()(prop_logits[pidx, loss_slice, :], targets).item() for pidx in range(prop_logits.shape[0])]
                new_loss = min(losses)
                min_idx = np.array(losses).argmin()
                if prev_loss is None or new_loss < prev_loss:
                    input_ids = proposals[min_idx]
                    prev_loss = new_loss
        return self.tokenizer.decode(input_ids)
