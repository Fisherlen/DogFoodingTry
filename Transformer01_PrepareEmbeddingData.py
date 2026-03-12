"""
Transformer01 - 跟着张老师手写transformer - 把消费者的问题转换为向量（input embedding + positional encoding)数据为机器学习做准备
                    By: Fisherlen Yu
                    Date: 2024/4/19
"""
import torch
import torch.nn as nn


print("""
===========================================================================================================================================
                                                        Part1   Prepare embedded data
===========================================================================================================================================
""")
# ---------------------------------------------------------- read data -----------------------------------------------------------------
# get the dataset to test - 打开梯子访问huggingface,找到datasets,搜索关键字: sales-text_book，再复制对应的数据LINK即可

# if not os.path.exists('sales-text_book.txt'):
#     url ='https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/resolve/main/sales_textbook.txt?download=true'
#     with open('data\\sales-text_book.txt', 'wb') as f:
#         f.write(requests.get(url).content)

with open('sales-text_book.txt', 'r') as f:
    text = f.read()


# ---------------------------------------------------------- setup hyperparameters -----------------------------------------------------
# 备注：第一个代码块将所有的超参数定义在一个字典 hyperparams 中,而第二个代码块则将每个超参数单独定义为一个变量。效果相同，根据个人喜好选择。
hyperparams = {
    'batch_size': 4,                                          # batch_size = 4  # How many batches per training step
    'block_size': 16,                                         # context_length = 16  # Length of the token chunk each batch
    'max_iters': 5000,                                        # max_iters = 5000  # Total of training iterations
    'eval_interval': 100,                                     # eval_interval = 50  # How often to evaluate the model
    'learning_rate': 3e-4,                                    # learning_rate = 1e-3  # 0.001
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 1337,                                             # TORCH_SEED = 1337
    'vocab_size': 50257,
    'n_embd': 64,                                             # d_model = 64  # The vector size of the token embeddings
    'n_head': 4,                                              # num_heads = 4  # Number of heads in Multi-head attention # 我们的代码中通过 d_model / num_heads = 来获取 head_size
    'n_layer': 8,                                             # num_layers = 8  # Number of transformer blocks
    'dropout': 0.1,                                           # dropout = 0.1 # Dropout rate
    'eval_iters': 20,                                         # eval_iters = 20  # How many iterations to average the loss over when evaluating the model
   }


# Hyperparameters 第二种超参定义方法（waylandzhang喜好的方式）

# batch_size = 4  # How many batches per training step
# context_length = 16  # Length of the token chunk each batch
# max_iters = 5000  # Total of training iterations
# eval_interval = 50  # How often to evaluate the model
# learning_rate = 1e-3  # 0.001
# device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Instead of using the cpu, we'll use the GPU if it's available.
# TORCH_SEED = 1337
# d_model = 64  # The vector size of the token embeddings
# num_heads = 4  # Number of heads in Multi-head attention # 我们的代码中通过 d_model / num_heads = 来获取 head_size
# num_layers = 8  # Number of transformer blocks
# dropout = 0.1 # Dropout rate
# eval_iters = 20  # How many iterations to average the loss over when evaluating the model

# torch.manual_seed(TORCH_SEED)

# ---------------------------------------------------------- tokenizer & split train data & validate data ------------------------------

import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")


tokenize_text = encoding.encode(text)
tokenize_text = torch.tensor(tokenize_text, dtype=torch.long)
max_token_value = tokenize_text.max().item()


train_index = int(len(tokenize_text) * 0.9)
train_data = tokenize_text[:train_index]
validate_data = tokenize_text[train_index:]
if __name__ == '__main__':
    print("""
    ------------------------------------------------------ tokenization ------------------------------------------------------
    """)
    print(f"\tThere are {len(tokenize_text)} tokens in the text")
    print(f"\tThe maximum token value is {max_token_value}")
    print(f"\tThere are {len(train_data)} tokens in the train data")
    print(f"\tThere are {len(validate_data)} tokens in the validate data")

data = train_data
idxs = torch.randint(low= 0, high= len(data) - hyperparams['block_size'], size= (hyperparams['batch_size'],))     # 初始化数据，随机抽取batch_size个数据
idxs
# print("""
# 备注：
# 每次运行该代码抽取出来的数据都不一样。
#     这是因为torch.randint()函数用于生成随机整数,其中low和high参数分别指定了随机整数的下限和上限(不包含上限)。
# 在给定的代码中,torch.randint(low=0, high=len(data) - hyperparams['block_size'], size=(hyperparams['batch_size'],))生成了一个形状为(hyperparams['batch_size'],)的一维张量,
# 其中每个元素都是一个在[0, len(data) - hyperparams['block_size'])范围内的随机整数。
# 由于随机数生成器每次运行时使用不同的种子,因此每次运行该代码时,生成的随机整数张量idex都会不同。所以,您每次运行该代码时得到的idex值都是不同的随机数据。
# """)

x_batch = torch.stack([torch.tensor(data[i:i+hyperparams['block_size']]) for i in idxs])
y_batch = torch.stack([torch.tensor(data[i+1:i+hyperparams['block_size']+1]) for i in idxs])   # y_batch = x_batch + 1 为了检验x_batch预测下一个值
x_batch.shape, y_batch.shape

import pandas as pd
df = pd.DataFrame(x_batch[0].numpy())
df
encoding.decode(x_batch[0].numpy())     # 把这批16个token都解码出来

# ---------------------------------------------------------- define input/word embedding table -----------------------------------------
input_embedding_lookup_table = nn.Embedding(num_embeddings=max_token_value, embedding_dim= hyperparams['n_embd'])
x_batch_embedding = input_embedding_lookup_table(x_batch)
y_batch_embedding = input_embedding_lookup_table(y_batch)
if __name__ == '__main__':
    print("""
    ------------------------------------------------------ define input embedding ------------------------------------------------------
    """)
    print('\t所有该文件data形成的向量矩阵维度：', input_embedding_lookup_table, '\n')
    print('\t所有该文件data张量化后形成的64维度矩阵，初始化信息 = 共100069行，64列（维）：\n\t', input_embedding_lookup_table.weight.data,'\n\n',)
    print('\tx_batch_embedding作为input embedding的形状：', x_batch_embedding.shape)
    print('\tx_batch_embedding作为input embedding的矩阵：\n\t', x_batch_embedding, '\n')
    print('\ty_batch输出相似，已阅，不再看')
    # print('\ty_batch_embedding作为input embedding的形状：', y_batch_embedding.shape, '\n')
    # print('\ty_batch_embedding作为input embedding的矩阵：\n', y_batch_embedding, '\n')

# ---------------------------------------------------------- define position encoding --------------------------------------------------
# define positional encoding 位置编码 （和前面的x_batch、y_batch进行相加，保持形状一致）

import math
position_encoding_lookup_table = torch.zeros(hyperparams['block_size'], hyperparams['n_embd'])  # 对该值初始化赋值为0
position = torch.arange(0, hyperparams['block_size'], dtype= torch.float).unsqueeze(1)
if __name__ == '__main__':
    print('''
    ------------------------------------------------------ define positional encoding ------------------------------------------------------
    ''')
    print('\t对position_encoding_lookup_table初始化赋值为0：\n\t', position_encoding_lookup_table.shape, '\n\t', position_encoding_lookup_table, "\n")
    print(f"\tposition初始化为空：{position}")

# apply sine and cosine to each position

div_term = torch.exp(torch.arange(0, hyperparams['n_embd'], 2).float() * (-math.log(10000.0)/hyperparams['n_embd']))
position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)    # 偶数位置 sine
position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)    # 奇数位置 cosine
position_encoding_lookup_table = position_encoding_lookup_table.unsqueeze(0).expand(hyperparams['batch_size'], -1, -1) # add batch to the first dimension

if __name__ == '__main__':
    print('\t对position_encoding_lookup_table赋值：\n\t', position_encoding_lookup_table.shape, '\n\t',
          position_encoding_lookup_table, "\n")
# ---------------------------------------------------------- define initial weights for starting machine learning ----------------------

# the initial input for machine learning ： add position encoding to the input embedding
x = x_batch_embedding + position_encoding_lookup_table
y = y_batch_embedding + position_encoding_lookup_table
if __name__ == '__main__':
    print("""
    ------------------------------------------------------ the initial input for machine learning: the value to be fed into the transformer block ----
    """)

    print('\tx的形状和张量矩阵：', x.shape, x)
    print('\ty的形状和张量矩阵：', y.shape, y, '\n')
    # pd.DataFrame(x[0].detach().numpy())
    print(pd.DataFrame(x[0].detach().numpy()))

    x_plot = x[0].detach().cpu().numpy()
    print("Final Input Embedding of x: \n", pd.DataFrame(x_plot))

X = x
Y = y

print('\n\n\n--> Part1 ENDED (value to be fed into the transformer block), going to Part2: Transformer Block')

