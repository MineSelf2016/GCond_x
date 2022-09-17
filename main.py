#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch 

from utils import *
from torch_geometric.utils import *
from easydict import EasyDict

import os.path as osp
from torch_geometric.datasets import Planetoid
from GCond import GCond

#%%
args = EasyDict(
    {
        "num_origin_nodes": 2708,
        "num_origin_features": 1433,
        "num_classes": 7,

        "num_syn_nodes": 100,
        "num_syn_features": 1433,
        "num_hidden_features": 128,
        "nlayers": 4,

        "reduction_ratio": 0.1,

        "device": "cpu"
    }
)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', "Cora")
dataset = Planetoid(path, "Cora")
graph = dataset[0]

# %%
x = graph.x
edge_index = graph.edge_index
y:torch.Tensor = graph.y

A =  to_dense_adj(edge_index).squeeze()

# %%
agent = GCond(graph.x, A, y, args)

