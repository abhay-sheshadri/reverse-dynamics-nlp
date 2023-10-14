#!/bin/bash

python stationary_reversal_loss.py --samples 1000 --suffix_batch_size 1 --sample_length 50 --vocab_batch_size 262 
python forward_llm_loss.py --samples 1000 --batch_size 1 --sample_length 50
python reverse_llm_loss.py --samples 1000 --batch_size 1 --sample_lenth 50
