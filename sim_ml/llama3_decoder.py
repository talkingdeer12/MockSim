import torch
import torch.nn as nn

class FakeLlama3DecoderBlock(nn.Module):
    def __init__(self, hidden_size, layer_idx=0):
        super().__init__()
        self.sim_layer_idx = layer_idx
        self.self_attn = nn.Linear(hidden_size, hidden_size)
        self.ffn1 = nn.Linear(hidden_size, 4*hidden_size)
        self.ffn2 = nn.Linear(4*hidden_size, hidden_size)

    def forward(self, x):
        h = self.self_attn(x)
        #h = torch.relu(h)
        #h = self.ffn1(h)
        #h = torch.relu(h)
        #h = self.ffn2(h)
        return h
