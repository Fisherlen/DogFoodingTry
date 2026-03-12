import torch
from Transformer05_1_model_中文小说 import TransformerLanguageModel

model = TransformerLanguageModel(max_token_value=4467) # 此处需改为模型保存时的最大token值和训练集的最大token值保持一致才能运行
# state_dict = torch.load("data\\model-scif.pt", map_location=torch.device('cpu'))  # CPU上运行使用此段代码
state_dict = torch.load("data\\model-scif-tiny.pt", map_location=torch.device('cpu'))  # CPU上运行使用此段代码
# state_dict = torch.load("data\\model-scif.pt") # 如果是运行在GPU上使用此段代码
model.load_state_dict(state_dict)

# step2: Calculate the number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {total_params:,} trainable parameters.')