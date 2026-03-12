"""
数据源：大模型课程：第一阶段《基础入门》|默认班级
第16课：准备数据集Data Loader
第17课：数据向量化矩阵化
第18课 ：Positional encoding
第19课：注意力机制
备注：使用的是OpenAI的tokenizer模型（英文版），数据集是中文的话，容易出现乱码
"""

print('''
# ---------------------------------- 1/ Prepare hyper parametres -------------------------
''')

import torch
batch_size = 4
context_length = 16
d_model = 512
num_heads = 8
device = "cuda" if torch.cuda.is_available() else "cpu"


print('''
# ---------------------------------- 2/ Prepare dataset ----------------------------------
''')

# # prepare  are data  from huggingface
# import os
# import requests
# if not os.path.exists("some_data.txt"):
#     download_url = ""
#     with open("some_data.txt", "w") as f:
#         f.write(requests.get(download_url).text)

with open("L:\\AI\\OpenAIProject\\AGI11_Transformer\\some_data.txt", "r") as f:
    text = f.read()     # 读取文本，原始数据保持不变
    # print(text)
    print(text[:1000])
    print('该TEXT一共有：', len(text)/10000,'万字')
    print('\n')


print('''
# ---------------------------------- 3/ Tokenization -------------------------------------
''')

# tokenization -- use tiktoken from OpenAI
import tiktoken
encoding = tiktoken.get_encoding('cl100k_base')
tokenized_text = encoding.encode(text)
# print('观察前10个token的编码：', tokenized_text[:10])
# print('观察上面编码对应的文字/符号：', encoding.decode(tokenized_text[:10]))
# 使用for循环显示每个token及其对应的编码
for i, token in enumerate(tokenized_text[:10]):
    print(f"Token {i+1}: 编码为 {token}, 对应的文字/符号为 '{encoding.decode([token])}'")

# 对比文字、token和有效token数量
print('\n')
total_text = len(text)
total_tokens = len(tokenized_text)
vocab_size = len(set(tokenized_text))
max_token = max(tokenized_text)
print(f"总字数：{total_text/10000}万")
print(f"总token数：{total_tokens/10000}万   备注：有的词被拆分为几个token，因而比总字数要多")
print(f"总词数：{vocab_size}个字词")
print(f"最大token编码：{max_token}")
print('\n')


#
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long,  device=device)
print(f'向量化后前 {len(tokenized_text[:10])}个token的编码依次是 ：{tokenized_text[:10]}')
print(tokenized_text.shape)
print('\n')


split_idx = int(0.8 * len(tokenized_text))
train_data = tokenized_text[:split_idx] # 80%的数据用于训练
val_data = tokenized_text[split_idx:]   # 20%的数据用于验证

# prepare x_batch, y_batch
idxs = torch.randint(low = 0, high = len(train_data) - context_length, size = (batch_size,))
x_batch = torch.stack([train_data[i:i+context_length] for i in idxs]) #从train_data中根据之前生成的索引idxs提取出一批具有context_length长度的数据片段
y_batch = torch.stack([train_data[i+1:i+context_length+1] for i in idxs])
print('x_batch未初始化的形状：', x_batch.shape, f'{batch_size}批具有{context_length}长度的数据片段','\ny_batch未初始化的形状：', y_batch.shape)
print('\n')

import pandas as pd
print('Our x batch as below:(未初始化的值)')
print(pd.DataFrame(x_batch))
print('Our y batch as below:(未初始化的值)')
print(pd.DataFrame(y_batch))
print('\n')

token_embedding_lookup_table = torch.nn.Embedding(max_token + 1, d_model, device = device)
x = token_embedding_lookup_table(x_batch)
y = token_embedding_lookup_table(y_batch)
print('初始化的x,y的形状：', x.shape, y.shape)
print('\n')

print('Our x as below:(初始化的值),context length as row, dimension as column')
print(pd.DataFrame(x[0].detach().cpu().numpy())) # 显示x的初始化值，第一批x[0]


print('''
# ---------------------------------- 4/ Positional encoding  -----------------------------
''')

# Positional encoding
import math
position_encoding_lookup_table = torch.zeros(context_length, d_model)  # 位置信息编码初始化为0
print('位置信息 初始化为0：', position_encoding_lookup_table)
position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1) #一个形状为(n,)的一维张量转换为形状为(n, 1)的二维张量
print('position:', position)

div_term = torch.exp((-math.log(10000.0) * torch.arange(0, d_model, 2, dtype=torch.float) / d_model))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term) # 偶数位置的位置信息编码，使用正弦函数
print('position_encoding_lookup_table[:, 0::2]', position_encoding_lookup_table[:, 0::2])
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term) # 奇数位置使用余弦函数
print('\nOur position_encoding_lookup_table as below:')
print(pd.DataFrame(position_encoding_lookup_table.detach().cpu().numpy()))
print('\n')
print('position_encoding_lookup_table形状：', position_encoding_lookup_table.shape)
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(batch_size, context_length, d_model)
print('\nOur position_encoding_lookup_table as below with batch size covered:', position_encoding_lookup_table.shape)
print('\n')

x = x + position_encoding_lookup_table
y = y + position_encoding_lookup_table

print('有了位置信息编码后的x形状:', x.shape, '\t\n', x)

# 可视化 visualization
import matplotlib.pyplot as plot
def visualize_pe(pe):
    plot.imshow(pe, aspect='auto')
    plot.title('Positional encoding')
    plot.xlabel('Encoding dimension')
    plot.ylabel('Position index')
    plot.colorbar()
    plot.show
visualize_pe(position_encoding_lookup_table[0].cpu().numpy())   # 图形解读：越接近1表示越相似
print('\n')


print('''
# ---------------------------------- 5/ Multi head attention -----------------------------
''')
import torch.nn as nn

# prepare Q,K,V weights square matrix(d_model)
Wq = nn.Linear(d_model, d_model)
Wk = nn.Linear(d_model, d_model)
Wv = nn.Linear(d_model, d_model)

Q = Wq(x)
K = Wk(x)
V = Wv(x)
print('Prepare Q K V as below')
print('\tQ,K,V初始化后形状（weighted）均一致：', Q.shape, K.shape, V.shape)

Q = Q.reshape(batch_size, context_length, num_heads, -1).permute(0,2,1,3)
K = K.reshape(batch_size, context_length, num_heads, -1).permute(0,2,1,3)
V = V.reshape(batch_size, context_length, num_heads, -1).permute(0,2,1,3)
print('\tQ、K、V把num_heads和context_length变换位置+切分多头之后的形状一样：', Q.shape, K.shape, V.shape)

 # Q @ K
attention = Q @ K.transpose(-2,-1)
print('\nPrepare attention as below:')
print('\t————> Attention1 Q @ K形状：', attention.shape)
print(pd.DataFrame(attention[0][0].detach().cpu().numpy()))

# scale
attention = attention / math.sqrt(d_model // num_heads)
print('\t————> Attention2 Q @ K/sqrt(d_model//num_heads)形状：', attention.shape)
print('留意与上一步数据大小对比（变小了）\n', pd.DataFrame(attention[0][0].detach().cpu().numpy()))

print('\n可视化看图 - 在Pycharm中无法查看；Jupyter中可看')
plot.imshow(attention[0][0].detach().cpu().numpy(),'Accent', aspect='auto')
plot.title('Attention : Q K')
plot.colorbar()

# mask
mask = torch.ones(attention.shape[-2:])
# print('初始化mask是1', mask.shape, '\n', mask)
mask = torch.triu(mask, diagonal=1)
# print('Create an upper triangular matrix with all values set to 1 and the rest set to 0', mask.shape, '\n',  mask)
mask = torch.triu(mask, diagonal=1).bool()
attention = attention.masked_fill(mask, float('-inf'))
print('\t————> Attention3 mask applied:\n', pd.DataFrame(attention[0][0].detach().cpu().numpy()))

print('\n可视化看图 - 在Pycharm中无法查看；Jupyter中可看')
plot.imshow(attention[0][0].detach().cpu().numpy(),'Accent', aspect='auto')
plot.title('Attention : Q K - masked_fill')
plot.colorbar()

# Probabilities - softmax
attention = torch.softmax(attention, dim=-1)  # 对最后一个维度dim=-1进行softmax
print('\t————> Attention4 概率化softmax\n', pd.DataFrame(attention[0][0].detach().cpu().numpy()))   # [0][0] 第一个批次第一个头

# @ V 回归最初形状
attention = attention @ V
print(f'\t————> Attention5 回归最初形状但还没合并{num_heads}个头\n', attention.shape, '\n', pd.DataFrame(attention[0][0].detach().cpu().numpy()))   # [0][0] 第一个批次第一个头

# concatenate heads
A = attention.transpose(1,2).reshape(batch_size, context_length,-1)  # -1是把d_model和num_heads做了合并列拼接
print('\t把多个头合并成X未分头时形状：', A.shape)
print(f'\t————> Attention6 多头合并后回归最初形状 - 第一个批次第一个字的{d_model}维度\n', pd.DataFrame(A[0][0].detach().cpu().numpy()))   # [0][0] 第一个批次第一个字的d_model维度坐标系

# output
Wo = nn.Linear(d_model, d_model)
output = Wo(A)

print('多头注意力的output形状和值如下：', output.shape)
print(f'某个token对应的{d_model}维度坐标系的位置:', A[0][2]) # 某个token对应的d_model坐标系的位置 ，第一个批次第三行所代表的token/文字


print('''
# ======================================= 如何打印参数 ======================================= 
''')

for name, value in Wo.named_parameters():
    print(name, value.shape, value)

