"""
Code_Challenge_I -
                    By: Fisherlen Yu
                    Date: 2024/5/6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import tiktoken
import pandas as pd

batch_size = 4
context_length = 16 # seq_length
d_model = 512
num_heads = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TORCH_SEED = 42
torch.manual_seed(TORCH_SEED)





"""
如何查看这个项目的路径所在位置
和上面代码无关
"""
import os

# 查看当前工作目录
print(os.getcwd())

# 查看项目根目录（如果有setup.py或pyproject.toml等标识文件）
import pathlib
project_root = pathlib.Path(__file__).parent.resolve()
print(project_root)



