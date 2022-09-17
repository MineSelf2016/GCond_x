#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn
import torch.nn.functional as F

from models.adjacent import adj_MLP
from utils import *
from easydict import EasyDict

# %%
class GCond():
    def __init__(self, origin_features, origin_A, origin_labels, args:dict) -> None:
        self.origin_features = origin_features
        self.origin_A = origin_A
        self.origin_labels = origin_labels
        self.num_origin_nodes = origin_features.shape[0]
        self.num_classes = args.num_classes

        self.device = args.device

        self.reduction_ratio = args.reduction_ratio
        self.num_syn_nodes = int(args.num_origin_nodes * self.reduction_ratio)
        
        assert self.num_syn_nodes >= self.num_classes, "The number of synthetic nodes is less than the number of classes"

        # num_syn_features == num_origin_features
        self.num_syn_features = origin_features



        self.num_hidden_features = args.num_hidden_features

        self.syn_features = None # shape: [num_syn_nodes, num_syn_features]
        self.syn_adj = None # shape: [num_syn_nodes, num_syn_nodes]
        self.syn_labels = None # shape: [num_syn_nodes]

        self.adj_mlp = adj_MLP(self.num_syn_features, self.num_hidden_features, out_features = 1, n_layers = args.nlayers)

        self.init()

        assert list(self.syn_features.shape) == [self.num_syn_nodes, self.num_syn_features], "shape of synthetic features is wrong."
        assert list(self.syn_adj.shape) == [self.num_syn_nodes, self.num_syn_nodes], "shape of synthetic adj is wrong."
        assert list(self.syn_adj.shape) == self.num_syn_nodes, "shape of synthetic labels is wrong."

    def init(self):
        self.init_features()
        self.init_adj()
        self.init_labels()

    def init_features(self):
        # randomly initialize the node features
        self.syn_features = torch.randn(self.num_syn_nodes, self.num_syn_nodes)

    def init_adj(self):
        self.syn_adj = self.adj_mlp(self.syn_features)

    def init_labels(self) -> torch.Tensor:
        cc = Counter(self.origin_labels)
        cc = {
            k: max(int(v / self.num_origin_nodes * self.num_syn_nodes), 1) for k, v in cc.items()
        }

        residue = sum(cc.values()) - self.num_syn_nodes
        sorted_classes = np.array(sorted(cc.items(), key = lambda x : x[1], reverse = True))
        sorted_classes[0][1] = sorted_classes[0][1] - residue

        assert sorted_classes[:, 1].sum() == self.num_syn_nodes, "The number of synthetic is wrong {} / {}.".format(sum(cc.values()), self.num_syn_nodes)

        arr = [[x[0]] * x[1] for x in sorted_classes]

        syn_labels = []
        for element in arr:
            syn_labels.extend(element)

        self.syn_labels = torch.tensor(syn_labels, device = self.device)


#%%
args = EasyDict(
    {
        "num_origin_nodes": 100,
        "num_origin_features": 3,
        "num_classes": 7,

        "num_syn_nodes": 10,
        "num_syn_features": 3,
        "num_hidden_features": 128,
        "nlayers": 4,

        "reduction_ratio": 0.1,

        "device": "cpu"
    }
)

# %%
args

#%%
import os.path as osp
from torch_geometric.datasets import Planetoid

path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', "Cora")

#%%
dataset = Planetoid(path, "Cora")

# %%
graph = dataset[0]

# %%
x = graph.x
edge_index = graph.edge_index
y = graph.y


# %%
agent = GCond(x[1], torch.rand([len(x[0]), len(x[0])]), y, args)

# %%
