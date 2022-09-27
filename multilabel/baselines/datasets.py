#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp

#%%
"""
Amazon dataset:

Nodes: 1598960
Edges: 132169734
Feature: 200
Classes: 107

"""

#%%
base_path = "/Users/mineself2016/study/codeSpace/VSCodeWorksapce/PythonWorkSpace/GCond_x"

features = np.load(osp.join(base_path, "datasets/amazon/feats.npy"))
labels = np.load(osp.join(base_path, "datasets/amazon/labels.npy"))

#%%
features.shape
labels.shape

#%%
adj_full = np.load(osp.join(base_path, "datasets/amazon/adj_full.npz"))

# %%
adj_full.files

# %%
type(adj_full["data"])

# %%
adj_full["indices"].shape

# %%
adj_full["indptr"].shape

# %%
