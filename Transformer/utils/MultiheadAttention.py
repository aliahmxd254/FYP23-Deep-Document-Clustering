import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiheadAttention(nn.Module):
    
    def __init__ (self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model      # d_model - dimension of model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask=False):
        batch_size, sequence_length, d_model = x.size()
        print(f"x.size(): {x.size()}")
        print(x)
        qkv = self.qkv_layer(x)
        print(f"qkv.size(): {qkv.size()}")
        print(f"{batch_size}, {sequence_length}, {self.num_heads}, {self.head_dim}")
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        print(f"q.size(): {q.size()}, k.size(): {k.size()}, v.size(): {v.size()}")
        values, attention = self._scaled_dot_product(q, k, v, mask)
        print(f"values.size(): {values.size()}, attention.size(): {attention.size()}")
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out

    
    def _scaled_dot_product(self, q, k, v, mask=False):
    
        def GenerateMask(scaled):
            mask = torch.full(scaled.size(), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            return mask
        
        d_k = q.size()[-1]
        scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
        if not mask:
            scaled += GenerateMask(scaled)
        attention = F.softmax(scaled, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention