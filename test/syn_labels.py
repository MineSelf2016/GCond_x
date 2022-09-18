#%%
import torch
from collections import Counter
from torch_geometric.utils import *
import numpy as np

from torch_geometric.nn import GCN

#%%
origin_labels = [0, 0, 2, 2, 0, 4, 4, 4, 4, 3, 1, 1, 5, 5, 5, 5]

#%%
num_classes = 6

#%%
num_origin_nodes = len(origin_labels)
num_syn_nodes = 9

assert num_syn_nodes >= num_classes, "The number of synthetic nodes is less than the number of classes"

#%%
cc = Counter(origin_labels)
cc = {
    k: max(int(v / num_origin_nodes * num_syn_nodes), 1) for k, v in cc.items()
}

residue = sum(cc.values()) - num_syn_nodes
sorted_classes = np.array(sorted(cc.items(), key = lambda x : x[1], reverse = True))
sorted_classes[0][1] = sorted_classes[0][1] - residue

assert sorted_classes[:, 1].sum() == num_syn_nodes, "The number of synthetic is wrong {} / {}.".format(sum(cc.values()), num_syn_nodes)

arr = [[x[0]] * x[1] for x in sorted_classes]

syn_labels = []
for element in arr:
    syn_labels.extend(element)

syn_labels = torch.tensor(syn_labels)


#%%
edge_index = torch.tensor([[0, 1, 1, 0, 2, 3],[0, 1, 2, 2, 3, 2]])

edge_index =  to_dense_adj(edge_index).squeeze()

edge_index
