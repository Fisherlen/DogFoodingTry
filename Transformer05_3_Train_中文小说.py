"""
备注：
    此部分训练目的=调通代码
    跑在CPU上，参数设置很低，训练效果很差（Training Loss:  Validation Loss: ）
"""

import torch
from Transformer05_1_model_中文小说 import TransformerLanguageModel


# Hyperparameters
batch_size = 4
context_length = 16  # Length of the token chunk each batch随机从测试集中随机抽取对应个数token参与训练计算weights
max_iterations = 1000
learning_rate = 1e-3    # 0.001
eval_interval = 50
eval_iters = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)


# # AIM Logs
# run = Run()
# run['hparams'] = {
#     'batch_size': batch_size,
#     'context_length': context_length,
#     'max_iterations': max_iterations,
#     'learning_rate': learning_rate,
#     'eval_interval': eval_interval,
#     'eval_iters': eval_iters,
#     'device': device,
#     'TORCH_SEED': TORCH_SEED,
# }


# 准备训练中文数据
with open('data/scifi.txt', 'r', encoding="utf-8") as file:
    text = file.read()


def token_num():
    global vocab, vocab_size, max_token_value
    vocab = sorted(list(set(text)))
    vocab_size = max_token_value = len(vocab)
    print(f'这段数据集共有: {vocab_size} 个不同的token/字')

if __name__ == '__main__':
    token_num()
    max_token_value

# 加密和解密，丢一个字符串，就能得到一个数字列表，反之亦然
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for char, idx in char2idx.items()}
encode = lambda x:[char2idx[char] for char in x]
decode = lambda idxs:''.join([idx2char[idx] for idx in idxs])
tokenized_text = torch.tensor(encode(text), dtype=torch.long)

# Split train and validation
train_size = int(len(tokenized_text) * 0.8)
train_data = tokenized_text[:train_size]
val_data = tokenized_text[train_size:]
# Initialize the model
# model = TransformerLanguageModel(max_token_value=vocab_size).to(device)
model = TransformerLanguageModel()
model = model.to(device)


# Get input embedding batch
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y



# calculate the loss@torch.no_grad( )
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch,y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train( )
    return out


#	Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iterations):
    if step % eval_iters == 0 or step == max_iterations - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step,
              'Training Loss:', round(losses['train'].item(), 3),
              'Validation Loss:', round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()










# Save the model
torch.save(model.state_dict(), 'data/model-scif-tiny.pt')

#
"""
来源：抖音，@LLM张老师，合集，第11集
备注：

"""



