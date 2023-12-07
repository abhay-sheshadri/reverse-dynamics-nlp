#!/bin/bash
date
python reversal_suffix_elicitation.py --model_size 160m --eval_size 250 --dataset pile_val --vocab_batch_size 1000
date
python reversal_suffix_elicitation.py --model_size 160m --eval_size 250 --dataset allenai/real-toxicity-prompts --vocab_batch_size 1000
date



#python reversal_suffix_elicitation.py --model_size 1.4B --eval_size 50
#python reversal_suffix_elicitation.py --model_size 2.8B --eval_size 50

#date
#python reversal_suffix_elicitation.py --model_size 1B --eval_size 50
#date
# old run
#date
#python reversal_suffix_elicitation.py --model_size 410m --eval_size 10
#date
#python reversal_suffix_elicitation.py --model_size 410m --eval_size 50
#date
#python reversal_suffix_elicitation.py --model_size 160m --eval_size 50
#date
