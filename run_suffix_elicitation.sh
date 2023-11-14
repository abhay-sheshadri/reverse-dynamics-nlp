#!/bin/bash
python reversal_suffix_elicitation.py --num_params 160 --eval_size 30 --dataset allenai/real-toxicity-prompts
python reversal_suffix_elicitation.py --num_params 410 --eval_size 30 --dataset allenai/real-toxicity-prompts
python reversal_suffix_elicitation.py --num_params 1B --eval_size 30 --dataset allenai/real-toxicity-prompts

