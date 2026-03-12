"""
Transformer - 跟着张老师手写transformer - 多头注意力机制+层归一化（layer normalizaiton） + 前馈神经网络 (feed forward network)
                    By: Fisherlen Yu
                    Date: 2024/4/23
"""
import pandas as pd
import torch
import numpy as np
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from Transformer01_PrepareEmbeddingData import hyperparams, X

print("""
===========================================================================================================================================
                                        Part2   Transformer Block
                        Calculate Attention Score (Q/K/V preparation, and algorithm to allocate score)
===========================================================================================================================================
""")

# 设置随机种子
def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    random.seed(seed_value)

# Now call the function to set the seed
set_seed(hyperparams['seed'])
    # 和超参设置的随机种子一致，确保每次运行，Q\K\V的初始化值，不会改变，但这3个变量的矩阵/张量值初始化的值会有不同


"""
    在Transformer模型中，Q（Query）、K（Key）、V（Value）是注意力机制（Attention Mechanism）的关键组成部分。
    这些矩阵是通过线性变换（Linear Transformation）从输入数据X中得到的，这个过程可以类比于在小学数学中学到的坐标变换。

    想象一下，你有一个画着很多点的图纸（输入数据X），现在你想用另一种方式来看这些点，比如通过放大镜（线性变换）来看。
    线性变换就像是这个放大镜，它帮助你以不同的方式来看这些点。在这个过程中，每个点都会移动到新的位置，这个新的位置就是通过线性变换得到的。
    
    在Transformer模型中，线性变换的作用是将输入数据X（可以看作是一系列的词嵌入）转换成新的表示形式Q、K、V。
    这些新的表示形式对于计算注意力分数非常重要，因为它们决定了模型在处理序列数据时的关注点。具体来说，nn.Linear是一个线性层，它执行了一个线性变换，这个变换可以表示为：
        Y = XA + b
            X 是输入数据（在Transformer中是词嵌入）
            A 是变换矩阵（在Transformer中是Wq、Wk、Wv）
            b 是偏置向量（在这个例子中设置为False，所以没有偏置）
            Y 是输出数据（在Transformer中是Q、K、V）
"""
# 初始化Q, K, V
# 创建三个线性层（Wq、Wk、Wv），每个层都会将输入数据X转换成新的表示形式Q、K、V。这些新的表示形式随后会被用于计算注意力分数，以便模型能够理解输入数据中的不同部分的重要性。
Wq = nn.Linear(hyperparams['n_embd'], hyperparams['n_embd'], bias= False)
Wk = nn.Linear(hyperparams['n_embd'], hyperparams['n_embd'], bias= False)
Wv = nn.Linear(hyperparams['n_embd'], hyperparams['n_embd'], bias= False)
Q = Wq(X)
K = Wk(X)
V = Wv(X)
if __name__ == '__main__':
    print("""
    ------------------------------------------------------ 1/2 get Q, K, V (same shape)--------------------------------------------
    """)
    print('\tX(initial embedding)的形状：', X.shape, '\n', '\t和张量矩阵:', X, '\n\n')
    print('\tQ的形状：', Q.shape, '\n', '\t和张量矩阵:', Q, '\n\n')
    print('\tK的形状：', K.shape, '\n', '\t和张量矩阵:', K, '\n\n')
    print('\tV的形状：', V.shape, '\n', '\t和张量矩阵:', V, '\n\n')




# apply multi head
Q = Q.reshape(hyperparams['batch_size'], hyperparams['block_size'], hyperparams['n_head'], hyperparams['n_embd'] // hyperparams['n_head']).permute(0, 2, 1, 3)
K = K.reshape(hyperparams['batch_size'], hyperparams['block_size'], hyperparams['n_head'], hyperparams['n_embd'] // hyperparams['n_head']).permute(0, 2, 1, 3)
V = V.reshape(hyperparams['batch_size'], hyperparams['block_size'], hyperparams['n_head'], hyperparams['n_embd'] // hyperparams['n_head']).permute(0, 2, 1, 3)
if __name__ == '__main__':
    print("""
    ------------------------------------------------------ 2/2 apply multi head ------------------------------------------------------
    """)
    print('Q的形状：', Q.shape, '\n', '\t和张量矩阵:', Q)
    print('K的形状：', K.shape, '\n', '\t和张量矩阵:', K)
    print('V的形状：', V.shape, '\n', '\t和张量矩阵:', V, '\n')


# 计算 Q 和 K 的 Attention 权重 & apply scale (2 steps combined in this one)

# 显式地加入了 Scaling 操作,这是 Transformer 模型中的一个常见技巧,可以帮助稳定训练过程
attention_score_3_QK = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hyperparams['n_embd'] // hyperparams['n_head'])
# output = Q @ K.transpose(-2, -1) / math.sqrt(hyperparams['n_embd'] // hyperparams['n_head']) 第二种方式
    # PyTorch 的 torch.matmul() 函数来计算矩阵乘法。不同的是第二种方式。
    # 备注：总的来说,这两种方式都正确，选择取决于偏好。如果你更喜欢使用 PyTorch 的内置函数,则可以选torch.matmul;如果你更喜欢简洁的语法,则可以选择第二种方式。

# apply mask
attention_score_5_mask = attention_score_3_QK.masked_fill(torch.triu(torch.ones(attention_score_3_QK.shape[-2:]), diagonal=1).bool(), float('-inf')) #[4, 4, 16, 16] [batch_size, num_heads, context_length, context_length]
pd.DataFrame(attention_score_5_mask[0,0].detach().cpu().numpy())

# apply softmax
attention_score_6_softmax = F.softmax(attention_score_5_mask, dim=-1)
# print(pd.DataFrame(attention_score.detach().cpu().numpy()))

# apply attention @ V
attention_score_7_V = torch.matmul(attention_score_6_softmax, V)    # 第一种方式，张老师在github上的表达。显式地加入了 Scaling 操作,这是 Transformer 模型中的一个常见技巧,可以帮助稳定训练过程
# attention_score = attention_score @ V                             # 第二种方式，张老师抖音的写法

# concatenate heads / Concatenate and Output
attention_score = attention_score_7_V.permute(0, 2, 1, 3).reshape(hyperparams['batch_size'], hyperparams['block_size'], hyperparams['n_embd'])
Wo = nn.Linear(hyperparams['n_embd'], hyperparams['n_embd'], bias= False)
output = Wo(attention_score)




print("""
===========================================================================================================================================
                                        Part2   Transformer Block
                                Residual Connection and Layer Normalization, Feed forward network
===========================================================================================================================================
""")


# add residual connection & layer normalization
output = output + X
layer_norm = nn.LayerNorm(hyperparams['n_embd'])
output = layer_norm(output)
if __name__ == '__main__':
    print('\t\n1/3 initial residual connection & layer normalization:\t', output.shape, '\t\n', output)


# feed forward network
output = nn.Linear(hyperparams['n_embd'], hyperparams['n_embd'] * 4)(output)
output = nn.ReLU()(output)
output = nn.Linear(hyperparams['n_embd'] * 4, hyperparams['n_embd'])(output)
output = torch.dropout(output, p=hyperparams['dropout'], train=True)
if __name__ == '__main__':
    print('\t\n2/3 feed forward network步骤的output:', output.shape)

# again, add residual connection & layer normalization
output2 = output + X
layer_norm = nn.LayerNorm(hyperparams['n_embd'])
output2 = layer_norm(output2)
if __name__ == '__main__':
    print('\t3/3 feed forward network + layer normalization:', output2.shape)

    print('\n\tfeed forward network这步的output:', output)
    print('\tfeed forward network + layer normalization后的output:', output2)

