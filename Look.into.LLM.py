"""
Look.into.LLM -
                    By: Fisherlen Yu
                    Date: 2024/5/7
"""
import torch

model_path = "model-ckpt.pt"
model = torch.load(model_path)

# 打印模型的结构和内容
def print_model_contents(model):
    if isinstance(model, dict):
        for key, value in model.items():
            print(f"{key}: {value}")
            # 对于模型参数，直接打印形状而不打印具体数值以免输出过长
            if isinstance(value, torch.Tensor):
                print(f"Shape: {value.shape}")
                print(f"Content: {value.data}")
                print()
            else:
                # 如果值不是张量，直接打印
                print(f'Value: {model[key]}')
    else:
        print(model)

if __name__ == '__main__':
    print_model_contents(model)
