import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForCausalLM)
from utils import *


class PromptOptimizer:
    """
    Implementation of Default GCG method, using the default
    settings from the paper (https://arxiv.org/abs/2307.15043v1)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        n_proposals: int = 128,
        n_epochs: int = 500,
        n_top_indices: int = 256,
        prefix_loss_weight: float = 0.25
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.n_proposals = n_proposals
        self.n_epochs = n_epochs
        self.n_top_indices = n_top_indices
        self.prefix_loss_weight = prefix_loss_weight

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
        loss_slice,
        temperature = 0
    ):
        # Sample random proposals
        proposals = []
        if temperature:
            logits = self.model(input_ids).logits
            probs = SOFTMAX_FINAL(logits**temperature) #from utils
        for p in range(self.n_proposals):
            if temperature:
                token_pos = np.random.randint(input_slice.start, input_slice.stop)
                rand_token = torch.multinomial(probs[0, token_pos, :], 1).item()
            else:
                token_pos = np.random.randint(input_slice.start, input_slice.stop)
                rand_token = np.random.choice(top_indices[token_pos])
            prop = input_ids.clone()
            prop[token_pos] = rand_token
            proposals.append(prop)
        return torch.stack(proposals)

    def optimize(
        self,
        initial_input,
        target_string,
        use_prefix_loss=True,
        temperature=0,
    ):
        # Parse input strings into tokens
        initial_inputs = self.tokenizer.encode(initial_input, return_tensors="pt")[0].cuda()
        initial_targets = self.tokenizer.encode(target_string, return_tensors="pt")[0].cuda()
        input_ids = torch.cat([initial_inputs, initial_targets], dim=0)
        input_slice = slice(0, initial_inputs.shape[0])
        target_slice = slice(initial_inputs.shape[0], input_ids.shape[-1])
        loss_slice = slice(initial_inputs.shape[0] - 1, input_ids.shape[-1] - 1)
        # Shifted input slices for prefix loss calculation
        shifted1 = slice(0, initial_inputs.shape[0] - 1)
        shifted2 = slice(1, initial_inputs.shape[0])
        # Optimize input
        prev_loss = None
        for i in range(self.n_epochs):
            # Get proposals for next string
            top_indices = self.calculate_restricted_subset(input_ids, input_slice, target_slice, loss_slice)
            proposals = self.sample_proposals(input_ids, top_indices, input_slice, target_slice, loss_slice, temperature=temperature)
            # Choose the proposal with the lowest loss
            with torch.no_grad():
                prop_logits = self.model(proposals).logits
                targets = input_ids[target_slice]
                losses = [nn.CrossEntropyLoss()(prop_logits[pidx, loss_slice, :], targets).item() for pidx in range(prop_logits.shape[0])]
                # Add a penalty for unlikely prompts that are not very high-likelihood
                if use_prefix_loss:
                    shifted_inputs = input_ids[shifted2]
                    prefix_losses = [nn.CrossEntropyLoss()(prop_logits[pidx, shifted1, :], shifted_inputs).item() for pidx in range(prop_logits.shape[0])]
                    losses = [losses[i] + self.prefix_loss_weight * prefix_losses[i] for i in range(len(losses))]
                # Choose next prompt
                new_loss = min(losses)
                min_idx = np.array(losses).argmin()
                if prev_loss is None or new_loss < prev_loss:
                    input_ids = proposals[min_idx]
                    prev_loss = new_loss
        return self.tokenizer.decode(input_ids)


class PromptOptimizerRMR(PromptOptimizer):
    """
    Uses Reverse Model with restricted logits to sample proposals
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        reverse_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        n_proposals: int = 128,
        n_epochs: int = 256,
        n_top_indices: int = 128
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            n_proposals=n_proposals,
            n_epochs=n_epochs,
            n_top_indices=n_top_indices,
        )
        self.reverse_model = reverse_model

    def sample_reverse_with_restrictions(
        self,
        initial_input,
        target_string
    ):
        # Sample from the reverse model in a way that only limits it to tokens 
        # with high gradients according to the forwards model
        initial_inputs = self.tokenizer.encode(initial_input, return_tensors="pt")[0].cuda()
        initial_targets = self.tokenizer.encode(target_string, return_tensors="pt")[0].cuda()
        input_ids = torch.cat([initial_inputs, initial_targets], dim=0)
        input_slice = slice(0, initial_inputs.shape[0])
        target_slice = slice(initial_inputs.shape[0], input_ids.shape[-1])
        loss_slice = slice(initial_inputs.shape[0] - 1, input_ids.shape[-1] - 1)
        # Optimize input
        top_indices = self.calculate_restricted_subset(input_ids, input_slice, target_slice, loss_slice)
        proposals = self.sample_proposals(input_ids, top_indices, input_slice, target_slice, loss_slice)
        return proposals

    def sample_proposals(
        self,
        input_ids,
        top_indices,
        input_slice,
        target_slice,
        loss_slice
    ):
        # Sample using RMR technique instead
        initial_inputs = input_ids[input_slice]
        initial_targets = input_ids[target_slice]
        return reverse_output(
            self.reverse_model.generate(
                torch.flip(initial_targets[None, :], (1,)),
                do_sample=True,
                early_stopping=False,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([
                    SampleTopTokens(
                        initial_inputs.shape[0],
                        initial_targets.shape[0],
                        top_indices
                )]),
                max_new_tokens=initial_inputs.shape[0],
                renormalize_logits=True,
                num_return_sequences=self.n_proposals,
            )
        )
