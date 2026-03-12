"""
Transformer01 - 跟着张老师手写transformer
                    By: Fisherlen Yu
                    Date: 2024/4/23
"""

import torch
import torch.nn as nn
from Transformer01_PrepareEmbeddingData import hyperparams, max_token_value, X
from Transformer02_TransformerBlock import output2
import pandas as pd

print("""
===========================================================================================================================================
                                        Part3   Iterations: Repeat steps above

ONE transformer block is done. To continue:
    We need to repeat the same process for the rest of the number of transformer blocks(hyperparams['n_layer']
    By having multiple blocks, the output is trained and being passed into next block as input X, 
    so after iterations the model can learn more complex patterns and relationships between the words in the input sequence.

===========================================================================================================================================
""")


print("""
===========================================================================================================================================
                                        Part4   Output Probabilities
===========================================================================================================================================
""")


# Apply the final linear layer to get the logits
logits = nn.Linear(hyperparams['n_embd'], max_token_value)(output2)
print(pd.DataFrame(logits[0].detach().cpu().numpy()))


# Then a final softmax function is used to convert the logits of the linear layer into a probability distribution.
probabilities = torch.softmax(logits, dim=-1)


