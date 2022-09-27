#%%
from tkinter import Y
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, confusion_matrix

#%%
num_o_data = 100
num_o_features = 10
num_o_classes = 5

num_s_data = 20
num_s_features = num_o_features
num_s_classes = 5

num_h_features = 1280

# %%
x_o = torch.randn(num_o_data, num_o_features)
y_o = torch.tensor([])
for i in range(20):
    y_o = torch.cat([y_o, torch.randperm(5)])

y_o = y_o.to(torch.long)

# %%
# make the dataset differentiable
x_s = nn.parameter.Parameter(torch.randn(num_s_data, num_s_features))
a_s = None
y_s = []
for i in range(5):
    y_s.extend([i] * 4)

y_s = torch.tensor(y_s, dtype = torch.long)

#%%
def match_loss(gw_o, gw_s, device = "cpu"):
    dist = torch.tensor(0.0).to(device)

    vec_o = []
    vec_s = []
    for i in range(len(gw_o)):
        vec_o.append(gw_o[i].reshape(-1))
        vec_s.append(gw_s[i].reshape(-1))

        dist += torch.sum(torch.pow((gw_o[i] - gw_s[i]), 2))
    # vec_o = torch.cat(vec_o, dim = 0)
    # vec_s = torch.cat(vec_s, dim = 0)

    # dist = torch.sum((vec_o - vec_s) ** 2)

    return dist

# %%
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features) -> None:
        super(MLP, self).__init__()
        self.linear_1 = nn.Linear(in_features, hidden_features)
        self.linear_2 = nn.Linear(hidden_features, hidden_features)
        self.linear_3 = nn.Linear(hidden_features, hidden_features)
        self.linear_4 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)
        x = F.relu(x)

        x = self.linear_3(x)
        x = F.relu(x)

        x = self.linear_4(x)
        x = F.softmax(x, dim = 1)

        return x 

#%%
model = MLP(num_o_features, num_h_features, num_o_classes)

#%%
def test_accuracy(model, input, labels: torch.Tensor):
    
    with torch.no_grad():
        output = model(input)

    predicted = torch.argmax(output, dim = 1)
    predicted = predicted.numpy()
    labels = labels.numpy()

    acc = accuracy_score(labels, predicted)

    # print("acc: ", acc)

    # cm = confusion_matrix(labels, predicted)
    # print(cm)
    return acc

#%%
def train():




    pass

#%%
loss_fn = nn.CrossEntropyLoss()
optim_model = torch.optim.SGD(model.parameters(), lr = 0.01)
optim_feature = torch.optim.SGD([x_s], lr = 0.01)

#%%
outer_loop = 100
inner_loop = 5

#%%
loss_grad_list = []
loss_class_list = []
for i in range(outer_loop):

    model_parameters = list(model.parameters())

    output_o = model(x_o)
    loss_o = loss_fn(output_o, y_o)
    gw_o = torch.autograd.grad(loss_o, model_parameters)

    output_s = model(x_s)
    loss_s = loss_fn(output_s, y_s)
    gw_s = torch.autograd.grad(loss_s, model_parameters, create_graph = True)

    # Update dataset
    loss_grad = match_loss(gw_o, gw_s)

    print(i, "th outer loop, loss", loss_grad.item())
    loss_grad_list.append(loss_grad.item())

    optim_feature.zero_grad()
    loss_grad.backward()
    optim_feature.step()

    # Update model
    for j in range(inner_loop):
        x_s_inner = x_s.detach()
        output_s_inner = model(x_s_inner)
        loss_s_inner = loss_fn(output_s_inner, y_s)

        if j == inner_loop - 1:
            print("\tclassification loss:", loss_s_inner.item())
            loss_class_list.append(loss_s_inner.item())
            print("accuracy on origin data: ", test_accuracy(model, x_o, y_o))
            print("accuracy on synthetic data: ", test_accuracy(model, x_s, y_s))

        optim_model.zero_grad()
        loss_s_inner.backward()
        optim_model.step()

#%%
plt.plot(loss_grad_list, label = "grad")
plt.plot(loss_class_list, label = "class")
plt.legend()

#%%
for i, layer in enumerate(gw_o):
    print(i, "th layer:")
    print("shape: ", layer.shape)
    print()

#%%
for i, layer in enumerate(gw_o):
    print(i, "th layer:")
    print("shape: ", layer.shape)
    print()

