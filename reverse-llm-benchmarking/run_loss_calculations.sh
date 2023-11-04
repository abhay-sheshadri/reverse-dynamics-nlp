#!/bin/bash
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288  --dist ../data/distributions/pile10k_empirical.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.2 --dist ../data/distributions/pile10k_empirical.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.4 --dist ../data/distributions/pile10k_empirical.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.2 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 0.4 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt
python stationary_reversal_loss.py --num_examples 100 --vocab_batch_size 6288 --dilution 1.0 --dist ../data/distributions/pythia-160m-deduped-v0_stationary_dist.pt



