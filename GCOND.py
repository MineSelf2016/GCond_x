#%%
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
from models.adjacent import adj_MLP

# %%
class GCond():
    def __init__(self, origin_features, origin_A, origin_labels, args:dict) -> None:
        self.origin_features = origin_features
        self.origin_A = origin_A
        self.origin_labels = origin_labels
        self.num_origin_features = self.origin_features.shape[1]
        self.num_origin_nodes = origin_features.shape[0]
        self.num_classes = args.num_classes

        self.device = args.device

        self.reduction_ratio = args.reduction_ratio
        self.num_syn_nodes = int(args.num_origin_nodes * self.reduction_ratio)
        
        self.lr_origin_task = 0.01
        self.lr_synthetic = 0.1

        assert self.num_syn_nodes >= self.num_classes, "The number of synthetic nodes is less than the number of classes"

        # num_syn_features == num_origin_features
        self.num_syn_features = self.num_origin_features

        self.num_hidden_features = args.num_hidden_features

        self.syn_features = None # shape: [num_syn_nodes, num_syn_features]
        self.syn_adj = None # shape: [num_syn_nodes, num_syn_nodes]
        self.syn_labels = None # shape: [num_syn_nodes]

        self.adj_mlp = adj_MLP(self.num_syn_features, self.num_hidden_features, out_features = 1, num_nodes = self.num_syn_nodes, num_features = self.num_syn_features, n_layers = args.nlayers)

        self.init()

        assert list(self.syn_features.shape) == [self.num_syn_nodes, self.num_syn_features], "shape of synthetic features is wrong."
        assert list(self.syn_adj.shape) == [self.num_syn_nodes, self.num_syn_nodes], "shape of synthetic adj is wrong."
        assert self.syn_labels.shape[0] == self.num_syn_nodes, "shape of synthetic labels is wrong. syn: {}, labels: {}".format(self.syn_labels.shape, self.num_syn_nodes)

    def init(self):
        self.init_features()
        self.init_adj()
        self.init_labels()

    def init_features(self):
        # randomly initialize the node features
        # use nn.Parameters to make it differentiable
        self.syn_features = nn.parameter.Parameter(torch.randn(self.num_syn_nodes, self.num_syn_features, device = self.device))

    def init_adj(self):
        self.syn_adj = self.adj_mlp(self.syn_features)

    def init_labels(self) -> torch.Tensor:
        cc = Counter(self.origin_labels.tolist())
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

    def forward_origin(self, x):
        pass 


    def forward_synthetic(self, x):
        pass 

    def contrast_gradient(self, x):
        pass 


