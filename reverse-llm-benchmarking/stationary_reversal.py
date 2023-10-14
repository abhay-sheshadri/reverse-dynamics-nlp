#%%
import numpy as np
from tqdm import tqdm
import time
import math
import gc
#%% 
import torch
from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, DataCollatorForLanguageModeling


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
def get_cond_logprob(input_ids, model):
    with torch.no_grad():
        logprobs = torch.nn.functional.log_softmax(model(input_ids=input_ids).logits, dim=-1)

    # Get the log probabilities corresponding to the words in input_ids
    relevant_logprobs = torch.gather(logprobs, 2, input_ids.unsqueeze(-1)[:, 1:]).squeeze(-1)
    
    # Sum log probabilities over the sequence length dimension
    sum_log_probs = relevant_logprobs.sum(dim=1)
  
    return sum_log_probs

def get_logprob(input_ids, model, stationary_dist):
  logprob = torch.log(stationary_dist[input_ids[:,0]])
  if input_ids.shape[1]>1:
     logprob = logprob + get_cond_logprob(input_ids[:,1:], model)
  return logprob


# Currently Incorrect 
def stationary_reverse_sample(
      model,
      stationary_dist, 
      prefix_length, 
      tokenized_suffix):
  
  model.eval()
  vocab_size = stationary_dist.shape[0]
  splus = tokenized_suffix
  for i in range(prefix_length):
    print("i=", i)
    psp = get_logprob(splus, model, stationary_dist)
    uniform_rv = torch.rand(1).item()
    j = 0 
    v0_sentence = torch.cat((torch.tensor([[0]]), splus), dim=-1)
    psp_given_v0 = get_cond_logprob(v0_sentence, model)

    c = torch.exp(psp_given_v0 + torch.log(stationary_dist[0]) - psp)

    pbar = tqdm(total=vocab_size)
    while c < uniform_rv:
        j += 1
        v_sentence = torch.cat((torch.tensor([[j]]), splus),dim=-1)
        psp_given_v = get_cond_logprob(v_sentence, model)
        newlogprob = psp_given_v + torch.log(stationary_dist[j]) - psp 
        c = c + torch.exp(newlogprob)
        if j % 1000 == 0:
          print(c)
          pbar.update(1000)
    pbar.close()
    p = torch.tensor([[j]])
    splus = torch.cat((p,splus),dim=-1)

# only supports batch_size 1 currently
def stationary_reverse_full_dist(
      model,
      stationary_dist, 
      prefix_length, 
      tokenized_suffix,
      vocab_batch_size=1572,
      renormalize_dist = True):
  
  model.eval()
  splus = tokenized_suffix
  vocab_size = stationary_dist.shape[0]
  vector_of_logprobs = torch.zeros(prefix_length, vocab_size)
  total_batches = math.ceil(vocab_size / vocab_batch_size)

  for i in range(prefix_length):
    print("i=", i)
    psp = get_logprob(splus, model, stationary_dist)
    for batch_num in tqdm(range(total_batches)):
            start_idx = batch_num * vocab_batch_size
            end_idx = start_idx + vocab_batch_size

            batch_indices = torch.arange(start_idx, end_idx).clamp(0, vocab_size-1).to(device)
            v_sentences = torch.cat((batch_indices.unsqueeze(1), splus.repeat(batch_indices.size(0), 1)), dim=-1)
            psp_given_v_batch = get_cond_logprob(v_sentences, model)


                # Check if it's the last batch
            if end_idx > vocab_size:
                # Calculate the actual batch size for the last batch
                actual_vocab_batch_size = vocab_size - start_idx
                # Create a tensor filled with the last value of stationary_dist of shape [vocab_batch_size]
                padded_stationary_dist = stationary_dist[-1].repeat(vocab_batch_size).to(device)
                # Replace the beginning of this tensor with the actual values from stationary_dist for this batch
                padded_stationary_dist[:actual_vocab_batch_size] = stationary_dist[start_idx:start_idx + actual_vocab_batch_size]
                log_stationary_dist = torch.log(padded_stationary_dist)
            else:
                log_stationary_dist = torch.log(stationary_dist[start_idx:end_idx])

            newlogprob_batch = psp_given_v_batch + torch.log(stationary_dist[start_idx:end_idx]) - psp 
            vector_of_logprobs[prefix_length - i - 1, start_idx:end_idx] = newlogprob_batch

            gc.collect()

    # Use softmax instead of .exp because vector_of_logprobs is not a distribution
    p = torch.distributions.Categorical(torch.nn.functional.softmax(vector_of_logprobs[prefix_length - i - 1,:])).sample()
    p = p.unsqueeze(0)
    p = p.unsqueeze(0).to(device)
    splus = torch.cat((p,splus),dim=-1)
    if renormalize_dist:
       vector_of_logprobs = torch.nn.functional.log_softmax(vector_of_logprobs,dim=-1)

  return vector_of_logprobs


# Currently batch size of 1 is the only supported batch size.
def stationary_reverse_full_dist_suffix_calculation(
      model,
      stationary_dist, 
      tokenized_suffix,
      vocab_batch_size=1572,
      renormalize_dist = True):
  
  model.eval()
  stationary_dist = stationary_dist.to(device)
  tokenized_suffix = tokenized_suffix.to(device)
  vocab_size = stationary_dist.shape[0]
  suffix_length = tokenized_suffix.shape[1]
  vector_of_logprobs = torch.zeros(suffix_length-1, vocab_size)

  for i in range(suffix_length-1):
      vector_of_logprobs[i,:] = stationary_reverse_full_dist(
          model,
          stationary_dist, 
          1, 
          tokenized_suffix[:,i+1:],
          vocab_batch_size=vocab_batch_size,
          renormalize_dist=renormalize_dist)[0,:]
      gc.collect()       

  return vector_of_logprobs

#%%

# Obama_log_probs_batch = stationary_reverse_full_dist(
#    model, empirical_dist, prefix_length, tokenized_suffix)
# # %%
# reverse_model = GPTNeoXForCausalLM.from_pretrained(
#   "afterless/reverse-pythia-160m"
# ).to(device)

# reverse_model.eval()
# rev_out = reverse_model(input_ids=tokenized_suffix)
# rev_logits = rev_out.logits
# rev_logprobs = torch.nn.functional.log_softmax(rev_logits, dim=-1)
# rev_probs = torch.exp(rev_logprobs)

# Obama_probs = torch.exp(Obama_log_probs)

# total_variation = torch.sum(torch.abs(Obama_probs[1,:] - rev_probs[:,:,1]))

# # %%
# # Check Pythia Conditional
# def get_conditional(
#       model,
#       tokenized_suffix,
#       vocab_batch_size=1572,
#       vocab_size=50304):
  
#   model.eval()
#   vector_of_logprobs = torch.zeros(1, vocab_size)
#   total_batches = math.ceil(vocab_size/vocab_batch_size)

#   for batch_num in tqdm(range(total_batches)):
#           start_idx = batch_num * vocab_batch_size
#           end_idx = start_idx + vocab_batch_size

#           batch_indices = torch.arange(start_idx, end_idx).clamp(0, vocab_size-1).to(device)
#           v_sentences = torch.cat((batch_indices.unsqueeze(1), tokenized_suffix.repeat(batch_indices.size(0), 1)), dim=-1)
#           print(v_sentences)
#           psp_given_v_batch = get_cond_logprob(v_sentences, model)

#           if end_idx > empirical_dist.shape[0]:
#               actual_vocab_batch_size = empirical_dist.shape[0] - start_idx
#               padded_empirical_dist = empirical_dist[-1].repeat(vocab_batch_size).to(device)
#               padded_empirical_dist[:actual_vocab_batch_size] = empirical_dist[start_idx:start_idx + actual_vocab_batch_size]
#               log_empirical_dist = torch.log(padded_empirical_dist)
#           else:
#               log_empirical_dist = torch.log(empirical_dist[start_idx:end_idx])

#           newlogprob_batch = psp_given_v_batch
#           vector_of_logprobs[0, start_idx:end_idx] = newlogprob_batch


#   return vector_of_logprobs
# # %%
# test_logprobs = check_conditional(model, tokenized_suffix)
# # %%

# from matplotlib import pyplot as plt

# test_probs = torch.exp(test_logprobs)
# sorted_probs = torch.sort(test_probs[0,:], descending=True)
# plt.figure(figsize=(10, 5))
# plt.plot(sorted_probs[0].numpy())
# plt.yscale('log')
# plt.xlabel("Token Rank")
# plt.ylabel("Probability")
# plt.title("Pythia Conditional Distribution")
# plt.show()



