
#%%
from gcn import GCN
# from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import os.path as osp

path = osp.join(osp.dirname(osp.realpath(__file__)), 'datasets', "Cora")
dataset = Planetoid(path, "Cora")
graph = dataset[0]


#%%
num_features = graph.num_features
edge_index = graph.edge_index
num_labels = dataset.num_classes
labels = graph.y
origin_datasets = graph.x


#%%
model = GCN(nfeat = num_features, nhid = 128, nclass = num_labels, nlayers = 4, device = "cpu")


#%%
model.parameters


  #%%
x = graph.x
edge_index = graph.edge_index

