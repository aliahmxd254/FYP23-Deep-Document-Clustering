import torch.nn as nn
from .EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads,
                                                   drop_prob) for _ in range(num_layers)])
    
    def forward(self, x):
        attention_list = []
        attention_matrix = None
        for layer in self.layers:
            attention_matrix = layer(x)
            attention_list.append(attention_matrix)
            x = attention_matrix
        return attention_matrix, attention_list
