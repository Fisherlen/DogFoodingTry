# Sample from trained model

import torch
from Transformer05_1_model_中文小说 import TransformerLanguageModel

# # Hyperparameters
# batch_size = 4 # How many batches per training step
# context_length= 16 # Length of the token chunk each batch
# eval_iters = 20# Number of iteratlons to average for evaluation
# max_iters = 500 # Total of training iterations <- Change this to smaller number for testing learning_rate = 1e-4# 0.001
# eval_interval = 50 # How often to evaluate
# eval_iters = 20# Number of iterations to average for evaluation
device = 'cuda' if torch.cuda.is_available() else 'cpu'     # Use GPU if it's available.
# TORCH_SEED = 1337
# torch.manual_seed(TORCH_SEED)

with open('data/scifi.txt', 'r', encoding="utf-8") as file:
    text = file.read( )

vocab = sorted(list(set(text)))# Called chars in the video, but vocab is a more generic term. Both are correct.
vocab_size = max_token_value =len(vocab)
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for char, idx in char2idx.items()}
encode = lambda x:[char2idx[char] for char in x]
decode = lambda idxs:''.join([idx2char[idx] for idx in idxs])
tokenized_text = torch.tensor(encode(text), dtype=torch.long)

# Initiate from trained model
model = TransformerLanguageModel(max_token_value=vocab_size).to(device)
model.load_state_dict(torch.load('data\\model-scif.pt', map_location=torch.device('cpu')))
model.eval()
start ='这个动作其实比被他狠狠操干还要令加文感到羞耻'
start_ids = encode(start)
x=(torch.tensor(start_ids, dtype=torch.long, device=device) [None,...])
y = model.generate(x, max_new_tokens=1000)
print('---------------')
print(decode(y[0].tolist()))
print('---------------')