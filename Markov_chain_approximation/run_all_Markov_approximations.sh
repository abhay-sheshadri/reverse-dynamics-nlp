#!/bin/bash
# Temporary: Currently batch size should be a factor of vocab_size
# Factors to try: {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 
# 64, 96, 128, 131, 192, 262, 384, 393, 524, 786, 1048, 
# 1572, 2096, 3144, 4192, 6288, 8384, 12576, 16768, 25152, 50304}


python Markov_chain_approximation.py  --min_prob 1e-3 --batch_size 1572 --model_name pythia-70m-deduped-v0
python Markov_chain_approximation.py --min_prob 1e-3 --batch_size 786 --model_name EleutherAI/pythia-160m-deduped-v0
python Markov_chain_approximation.py --min_prob 1e-3 --batch_size 786 --model_name EleutherAI/pythia-410m-deduped-v0

