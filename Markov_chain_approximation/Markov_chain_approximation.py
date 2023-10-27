#%%
import argparse
import numpy as np
import scipy.sparse as sps
from scipy.special import logsumexp
from tqdm import tqdm
import math
import os
import gc
import torch
from transformers import GPTNeoXForCausalLM
# import einsum
#%%

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
  err_vec = torch.zeros(vocab_size).to(device)

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

        err_vec[start_idx:end_idx] = 1-partial_sums.squeeze()

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
  return transition_matrix, err_vec


def log_space_product(A,B):
    Astack = np.stack([A]*B.shape[1]).transpose(1,0,2)
    Bstack = np.stack([B]*A.shape[0]).transpose(0,2,1)
    return logsumexp(Astack+Bstack, axis=2)

def model_left_multiply(model, 
                        distribution, 
                        device,
                        batch_size=1572,
                        vocab_size=50304,
                        logspace=False):

  assert vocab_size % batch_size == 0

  total_batches = math.ceil(vocab_size/batch_size)
  if logspace == True:
    distribution = torch.log(distribution)
  

  for batch_num in tqdm(range(total_batches)):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        input_ids = torch.arange(start_idx, end_idx).clamp_(0, vocab_size-1).unsqueeze(1)
        input_ids = input_ids.to(device)

        with torch.no_grad():
          outputs = model(input_ids=input_ids)
          logits = outputs.logits.float()
          logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
          if logspace == False:
            probs = torch.exp(logprobs)

        # logprobs.shape is (batch_size, 1, vocab_size)
        # distribution.shape is 
        if logspace:
          out_vec += torch.tensor(log_space_product(distribution[start_idx:end_idx].unsqueeze(0),logprobs.squeeze(1))).squeeze(0)
        else:
          out_vec += distribution[start_idx:end_idx] @ probs.squeeze(1)

  return out_vec

def model_left_multiply_in_place(model, 
                        distribution, 
                        out_vec,
                        device,
                        batch_size=1572,
                        vocab_size=50304,
                        logspace=False):
  assert vocab_size % batch_size == 0
  total_batches = math.ceil(vocab_size/batch_size)
  if logspace == True:
    distribution = torch.log(distribution)
  
  for batch_num in tqdm(range(total_batches)):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        input_ids = torch.arange(start_idx, end_idx).clamp_(0, vocab_size-1).unsqueeze(1)
        input_ids = input_ids.to(device)

        with torch.no_grad():
          outputs = model(input_ids=input_ids)
          logits = outputs.logits.float()
          logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
          if logspace == False:
            probs = torch.exp(logprobs)

        if logspace:
          out_vec += torch.tensor(log_space_product(distribution[start_idx:end_idx].unsqueeze(0),logprobs.squeeze(1))).squeeze(0)
        else:
          out_vec += distribution[start_idx:end_idx] @ probs.squeeze(1)

  return                           

def model_left_power_iteration(model, 
                        distribution, 
                        device,
                        batch_size=1572,
                        vocab_size=50304,
                        logspace=False,
                        maxiter=1000,
                        tol=1e-8):
   
  assert vocab_size % batch_size == 0
  out = torch.copy(distribution)
  out_plus = torch.zeros(vocab_size).to(device)
  for i in tqdm(range(1, maxiter)): 
     model_left_multiply_in_place(model, out, out_plus, device, batch_size, vocab_size)
     err = torch.abs(out_plus - out).sum()
     print(err)
     if err < tol:
       return out_plus
     out[:]=out_plus   
  print("Failed to converge before maxiter.")
  return out

#%%
def is_stochastic_vector(pi, dim=0,tol=1e-8):
  if abs(pi.sum(dim=dim)-1)>tol:
    return False
  if not (pi>=0).all(dim=dim):
    return False
  return True


def is_distribution(tensor, dim=-1, tol=1e-5):
    """
    Checks if slices along the specified dimension of the tensor are distributions.

    Parameters:
    - tensor: input numpy ndarray or pytorch tensor.
    - dim: dimension along which to check if slices are distributions.
    - tol: tolerance for checking if sum is close to 1.

    Returns:
    - boolean tensor (or ndarray) indicating which slices are distributions.
    """

    # Check if it's a PyTorch tensor
    if hasattr(tensor, 'is_cuda'):
        # Ensure the tensor is on CPU
        tensor = tensor.cpu()
        
        total = tensor.sum(dim=dim)
        is_non_negative = (tensor >= 0).all(dim=dim)
        
        return (is_non_negative) & (abs(total - 1.0) <= tol)

    # Assuming it's a numpy array otherwise
    else:
        total = tensor.sum(axis=dim)
        is_non_negative = (tensor >= 0).all(axis=dim)
        
        return (is_non_negative) & (abs(total - 1.0) <= tol)


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

#%%
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

    P = estimate_transition_matrix(model, device, batch_size=batch_size, filter_prob = min_prob)[0]
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
