from reverse_sampling import sample_reverse_dynamics_reverse_prior
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTNeoXForCausalLM.from_pretrained(
    "EleutherAI/pythia-12B-deduped",
).to(device)
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

reverse_model = GPTNeoXForCausalLM.from_pretrained(
    "afterless/reverse-pythia-160m"
).to(device)

tokenized_suffix = tokenizer.encode(" Clinton")


# current time
start = time.time()
out = sample_reverse_dynamics_reverse_prior(
    model,
    reverse_model,
    10,
    tokenized_suffix,
    vocab_batch_size=3000,
    temperature=1.0,
    dilution=0.0,
    device="cuda"
)
print(out)
end = time.time()
print(end - start)
tokenized_suffix = tokenizer.encode(" basketball")


# current time
start = time.time()
out = sample_reverse_dynamics_reverse_prior(
    model,
    reverse_model,
    10,
    tokenized_suffix,
    vocab_batch_size=3000,
    temperature=1.0,
    dilution=0.0,
    device="cuda"
)
print(out)
end = time.time()
print(end - start)






