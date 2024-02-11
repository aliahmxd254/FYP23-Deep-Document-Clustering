import torch.nn as nn
from utils.EncoderLayer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads,
                                                   drop_prob) for _ in range(num_layers)])
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
