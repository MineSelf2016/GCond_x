#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
num_nodes = 4
num_features = 3
hidden_features = 128

x = torch.rand(num_nodes, num_features)
A = torch.empty(num_nodes, num_nodes)

# %%
class adj_MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_layers = 4) -> None:
        super(adj_MLP, self).__init__()
        self.linear_1 = nn.Linear(in_features * 2, hidden_features)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_features)
        self.linear_2 = nn.Linear(hidden_features, hidden_features)
        self.batch_norm_2 = nn.BatchNorm1d(hidden_features)
        self.linear_3 = nn.Linear(hidden_features, hidden_features)
        self.batch_norm_3 = nn.BatchNorm1d(hidden_features)
        self.linear_4 = nn.Linear(hidden_features, out_features)

        self.reset_param()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a_l = x.repeat(1, num_nodes).reshape(num_nodes, num_nodes, num_features)

        a_r = x.repeat(num_nodes, 1).reshape(num_nodes, num_nodes, num_features)

        a_syn = torch.cat([a_l, a_r], dim = 2) # shape: [num_nodes, num_nodes, num_features * 2]

        a_syn = self.linear_1(a_syn)
        # a_syn = self.batch_norm_1(a_syn)
        a_syn = F.relu(a_syn)

        a_syn = self.linear_2(a_syn)
        # a_syn = self.batch_norm_2(a_syn)
        a_syn = F.relu(a_syn)

        a_syn = self.linear_3(a_syn)
        # a_syn = self.batch_norm_3(a_syn)
        a_syn = F.relu(a_syn)

        # at the last layer, there is no activation and batch_norm
        a_syn = self.linear_4(a_syn) # shape: [num_nodes, num_nodes, 1]

        a_syn = torch.squeeze(a_syn, 2) # shape: [num_nodes, num_nodes]

        a_syn = torch.add(a_syn, a_syn.T) / 2 # symmetric, ele-wise addition
        a_syn = torch.sigmoid(a_syn)

        return a_syn
    
    def reset_param(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                print("reset params")
            if isinstance(m, nn.BatchNorm1d):
                # nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                m.reset_parameters()

        self.apply(weight_reset)


#%%
adj_mlp = adj_MLP(num_features, hidden_features, 1)

# %%
a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]], dtype = torch.float)

#%%
output:torch.Tensor = adj_mlp(a)

print(output.shape)
print(output)

#%%
