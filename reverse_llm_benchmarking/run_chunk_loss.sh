prefix_length=20
num_examples=50
filename_prefix=al_suffix_20_chunk_test_early_doc_bias
dataset_name=pile_val
seed=1776
device=cuda
full_data_set_chunk=true
vocab_batch_size=500

python all_methods_loss.py --num_examples $num_examples --prefix_length $prefix_length --reverse_model_prior true --model_size "160m" --filename_prefix $filename_prefix  --dataset_name $dataset_name --seed $seed --device $device --full_data_set_chunk $full_data_set_chunk --vocab_batch_size $vocab_batch_size

