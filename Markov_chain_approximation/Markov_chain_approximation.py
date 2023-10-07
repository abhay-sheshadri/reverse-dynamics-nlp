#%%
import argparse
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm
import math
import os
import gc
import torch
from transformers import GPTNeoXForCausalLM

def parse_arguments():
  parser = argparse.ArgumentParser(description="Process some arguments.")

  parser.add_argument('--min_prob', type=float, default=1e-3,
                      help='Number of samples to keep.')

  parser.add_argument('--batch_size', type=int, default=1572,
                      help='Batch size for training.')
  
  parser.add_argument('--model_name', type=str, default = 'pythia-160m-deduped-v0',
                      help ='LLM model name.')

  return parser.parse_args()

#%%
def estimate_transition_matrix(
        model,
        device,
        batch_size=1572,
        vocab_size=50304,
        filter_prob=1e-3):
  
  # Temporary: Eventually allow for arbitrary batch_size
  # Factors to try: {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 
  # 64, 96, 128, 131, 192, 262, 384, 393, 524, 786, 1048, 
  # 1572, 2096, 3144, 4192, 6288, 8384, 12576, 16768, 25152, 50304}

  assert vocab_size % batch_size == 0

  total_batches = math.ceil(vocab_size/batch_size)
  indices_list = []
  values_list = []  

  filter_prob_tensor = torch.tensor(filter_prob).to(device) 

  for batch_num in tqdm(range(total_batches)):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        input_ids = torch.arange(start_idx, end_idx).clamp_(0, vocab_size-1).unsqueeze(1)
        input_ids = input_ids.to(device)

        with torch.no_grad():
          outputs = model(input_ids=input_ids)
          logits = outputs.logits.float()
          logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
          probs = torch.exp(logprobs)

        
        filter_mask = (probs < filter_prob_tensor)  
        filtered_probs = probs.masked_fill(filter_mask, 0.0)  
        ids, _, vocab_indices = torch.where(filtered_probs > 0)


        partial_sums = filtered_probs.sum(dim=-1, keepdim=True)
        normalized_probs = filtered_probs / partial_sums

        # print(sum(normalized_probs, dim=-1))

        values_list.append(normalized_probs[ids, 0, vocab_indices])
        ids += start_idx
        ids = ids.to(torch.int32)
        vocab_indices = vocab_indices.to(torch.int32)
        indices_list.append(torch.stack([ids, vocab_indices]))

        del logprobs, filter_mask, outputs
        torch.cuda.empty_cache()
        gc.collect()


  indices = torch.cat(indices_list, dim=1)
  values = torch.cat(values_list, dim=0)
  transition_matrix = torch.sparse_coo_tensor(indices, values, (vocab_size, vocab_size), device=device)
  return transition_matrix
#%%
def is_stochastic_vector(pi, tol=1e-8):
  if np.abs(np.sum(pi)-1)>tol:
    return False
  if not np.all(pi>=0):
    return False
  return True

def is_stochastic_matrix(W, tol = 1e-8):
  if sps.issparse(W):
   return np.all(np.abs((W@np.ones(W.shape[0]))-1)<tol) and np.all(W.data >=0)
  else:
    return np.all(np.abs(np.sum(W, 1)-1)<tol) and np.all(W >= 0)


# Q should be a rate matrix P-I for row stochastic P 
def compute_stationary_distribution(Q):
  sparse = sps.issparse(Q)
  if sparse == True:
    print("Using sparsity")
    Q=Q.T
    n = Q.shape[0]
    B = Q[0:n-1,0:n-1]
    d = Q[0:n-1,n-1]
    pi = np.array(sps.linalg.spsolve(B, -d))
    pi = np.squeeze(pi)
    pi = np.concatenate([pi,np.array([1.])])
    pi = pi/np.sum(pi)
  else:
    print("Not using sparsity")
    Q = Q.T
    n = Q.shape[0]
    B = Q[0:n-1,0:n-1]
    d = Q[0:n-1,n-1]
    pi = np.array(np.linalg.solve(B, -d))
    pi = np.squeeze(pi)
    pi = np.concatenate([pi,np.array([1.])])
    pi = pi/np.sum(pi)

  if not is_stochastic_vector(pi):
    print('The computed stationary distribution is not stochastic.')
    return pi

  return pi

def torch_sparse_to_scipy_csr(tensor: torch.Tensor) -> sps.csr_matrix:
    # Ensure the tensor is a sparse tensor
    assert tensor.is_sparse, "Input tensor must be sparse"
    
    # Convert the tensor components to numpy
    row_indices = tensor.indices()[0].cpu().numpy()
    col_indices = tensor.indices()[1].cpu().numpy()
    values = tensor.values().cpu().numpy()
    
    # Use the extracted values to construct a scipy CSR matrix
    csr_matrix = sps.csr_matrix((values, (row_indices, col_indices)), shape=tensor.shape)
    
    return csr_matrix

def main():

    args = parse_arguments()
    min_prob = args.min_prob
    batch_size = args.batch_size
    model_name = args.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/"+model_name
        ).to(device)

    vocab_size = model.config.vocab_size

    P = estimate_transition_matrix(model, device, batch_size=batch_size, filter_prob = min_prob)
    P_csr = torch_sparse_to_scipy_csr(P.coalesce())

    directory = "data/"+model_name
    if not os.path.exists(directory):
      os.makedirs(directory)

    formatted_number = "{:.1e}".format(min_prob)
    formatted_number = formatted_number.replace(".0e", "e")

    sps.save_npz(directory+"/"+"transition_matrix_minprob_" + formatted_number, P_csr)

    pi = compute_stationary_distribution(P_csr-sps.eye(vocab_size))
    np.save(directory+"/"+"stat_dist_minprob"+formatted_number, pi)


if __name__ == "__main__":
  main()

# %%
