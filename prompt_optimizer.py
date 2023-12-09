import numpy as np
import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GPTNeoXForCausalLM)
from utils import *
from reverse_llm_benchmarking.reverse_sampling import *


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
        n_epochs: int = 100,
        n_top_indices: int = 128,
        prefix_loss_weight: float = 0.0
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
        temperature = None
    ):
        # Sample random proposals
        proposals = []
        if temperature:
            logits = self.model(input_ids.view(1, *input_ids.shape)).logits
            probs = SOFTMAX_FINAL(logits/temperature) #from utils
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


class ReverseModelSampler:
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        reverse_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):

        self.model = model
        self.reverse_model = reverse_model
        self.tokenizer = tokenizer

    def optimize(
        self,
        initial_input,
        target_string,
        temperature=0,
    ):
        # Tokenize prefix and suffix
        prefix_tokens = self.tokenizer.encode(initial_input)
        suffix_tokens = self.tokenizer.encode(target_string)
        # Beam search
        prefix_list = reverse_normalized_beam_generate(
            self.reverse_model,
            self.tokenizer,
            target_string,
            len(prefix_tokens),
            beam_size=50
        )
        pairs_batch = torch.stack(prefix_list)
        pairs_batch = torch.cat((pairs_batch, torch.tensor([suffix_tokens]*len(prefix_list))), dim=1)
        # Call the batched loss function
        predicted_prefix_loss_batch, predicted_suffix_loss_batch = forward_loss_batch(
            self.model,
            pairs_batch,
            self.tokenizer,
            prefix_len=len(prefix_tokens)
        )        
        best_prefix = prefix_list[torch.argmin(predicted_suffix_loss_batch)]
        return self.tokenizer.decode(best_prefix.tolist() + suffix_tokens)
        """
        # Parse input strings into tokens
        initial_targets = reverse_tokenize(self.tokenizer, target_string)
        initial_inputs = self.tokenizer.encode(initial_input, return_tensors="pt").cuda()
        
        # Sample from the reverse model
        output = self.reverse_model.generate(
            initial_targets,
            max_new_tokens=initial_inputs.shape[-1],
            num_beams=50,
            num_return_sequences=1,
        )
        return reverse_decode(self.tokenizer, output)[0]
        """


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
        batch_size=1024
    ):

        self.model = model
        self.dist = dist
        self.tokenizer = tokenizer
        self.batch_size = batch_size


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
            device="cuda"
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


class AutoDAN:
    """
    AutoDAN c.f. https://openreview.net/pdf?id=rOiymxm8tQ
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch: int = 256,
        max_steps: int = 500,
        weight_1: float = 3,
        weight_2: float = 100,
        temperature: float = 1,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.batch = batch
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        self.temperature = temperature
        self.max_steps = max_steps


    def sample_model(
        self,
        input_ids,
    ):
        logits = self.model(input_ids).logits[:, -1, :]
        probs = SOFTMAX_FINAL(logits/self.temperature)
        samples = torch.multinomial(probs, 1)
        return samples


    def optimize(
        self,
        user_query,
        num_tokens,
        target_string,
        stop_its=1,
        verbose=False,
    ):
        '''
        Use stop_its to avoid early stopping with small batch sizes
        Note BOS token and custom dialogue templates not implemented yet
        Possible problem: tokenization iteratively adds tokens as characters, but these may be tokenized differently after addition
        '''
        query = self.tokenizer.encode(user_query, return_tensors="pt")[0].cuda()
        query_len = query.shape[-1]
        targets = self.tokenizer.encode(target_string, return_tensors="pt")[0].cuda()
        batch_targets = targets.unsqueeze(0).repeat(self.batch,1).contiguous()

        initial_x = self.sample_model(query.unsqueeze(0))[0]
        input_ids = torch.cat([query, initial_x, targets], dim=0)
        adversarial_sequence = []
        adversarial_seq_tensor = torch.tensor(adversarial_sequence,dtype=torch.long).cuda()

        for ind in range(num_tokens): #iteratively construct adversarially generated sequence
            curr_token = query_len + ind
            optimized_slice = slice(curr_token, curr_token+1)
            target_slice = slice(curr_token + 1, input_ids.shape[-1])
            loss_slice = slice(curr_token, input_ids.shape[-1] - 1)
            # print(f"slices: optimized slice {optimized_slice}, target slice {target_slice}, loss slice {loss_slice}")
            best_tokens = set()
            if verbose:
                print(f"For seq #{self.tokenizer.decode(input_ids)}#")
                print(f"Optimizing token {ind} at index {curr_token}: {self.tokenizer.decode(input_ids[curr_token])}")
            stop = 0
            for step in range(self.max_steps): #optimize current token
                grads, logits = token_gradients_with_output(self.model, input_ids, optimized_slice, target_slice, loss_slice)
                curr_token_logprobs = LOGSOFTMAX_FINAL(logits[0, curr_token-1, :])
                candidate_tokens = torch.topk(-1*self.weight_1*grads+curr_token_logprobs, self.batch-1, dim=-1).indices.detach()
                candidate_tokens = torch.cat((candidate_tokens[0],input_ids[curr_token:curr_token+1]),dim=0) #append previously chosen token
                candidate_sequences = input_ids.unsqueeze(0).repeat(self.batch,1).contiguous()
                candidate_sequences[:,curr_token] = candidate_tokens
                with torch.no_grad():
                    all_logits = self.model(candidate_sequences).logits
                    loss_logits = all_logits[:, loss_slice, :].contiguous()
                    target_losses = CROSSENT(loss_logits.view(-1,loss_logits.size(-1)), batch_targets.view(-1)) #un-reduced cross-ent
                    target_losses = torch.mean(target_losses.view(self.batch,-1), dim=1) #keep only batch dimension
                    combo_scores = -1*self.weight_2*target_losses + curr_token_logprobs[candidate_tokens]
                    combo_probs = SOFTMAX_FINAL(combo_scores/self.temperature)
                    temp_token = candidate_tokens[torch.multinomial(combo_probs, 1)]
                    best_token = candidate_tokens[torch.argmax(combo_probs).item()]
                if step == 0 and verbose:
                    print(f"max prob {torch.max(combo_probs):.2f} temp_token {self.tokenizer.decode(temp_token)} token_id {temp_token.item()} and best_token {self.tokenizer.decode(best_token)} token_id {best_token}")
                    print(f"10 candidate tokens at step {step}: {self.tokenizer.decode(candidate_tokens[:10])}")
                    print("Losses:")
                    print(f"     Initial loss at step {ind}, iteration {step}: {torch.max(-1*target_losses).item():.2f}")
                    print(f"     Combination scores at step {ind}, iteration {step}: {[round(val.item(),2) for val in torch.topk(combo_scores,5)[0]]}") #torch.topk(combo_scores,5)
                elif best_token in best_tokens:
                    stop+=1
                    if stop==stop_its:
                        if verbose:
                            print("Losses:")
                            print(f"     Final loss at step {ind}, iteration {step}: {torch.max(-1*target_losses).item():.2f}")
                            print(f"     Combination scores at step {ind}, iteration {step}: {[round(val.item(),2) for val in torch.topk(combo_scores,5)[0]]}") #torch.topk(combo_scores,5)
                            print(f"Best token {self.tokenizer.decode(best_token)} Sampled token {self.tokenizer.decode(temp_token)}")
                        adversarial_sequence.append(temp_token)
                        break
                else: 
                    best_tokens.add(best_token)
                if step==self.max_steps-1:
                    if verbose:
                        print("Losses:")
                        print(f"     Final loss at step {ind}, iteration {step}: {torch.max(-1*target_losses).item():.2f}")
                        print(f"     Combination scores at step {ind}, iteration {step}: {[round(val.item(),2) for val in torch.topk(combo_scores,5)[0]]}") #torch.topk(combo_scores,5)
                        print(f"Best token {self.tokenizer.decode(best_token)}. Sampled token {self.tokenizer.decode(temp_token)}.")
                    adversarial_sequence.append(temp_token)
                    break
                input_ids = torch.cat([query, adversarial_seq_tensor, temp_token, targets], dim=0)

            adversarial_seq_tensor = torch.tensor(adversarial_sequence,dtype=torch.long).cuda()
            next_in = torch.cat([query, adversarial_seq_tensor],dim=0).unsqueeze(0).type(torch.long)
            next_tok_rand = self.sample_model(next_in)[0]
            input_ids = torch.cat([query, adversarial_seq_tensor, next_tok_rand, targets], dim=0)


        # print('Final target logprob was:', torch.max(-1*target_losses).item())
        return self.tokenizer.decode(torch.tensor(adversarial_sequence))
