#%%
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
# make the dataset differential
x_s = nn.parameter.Parameter(torch.randn(num_s_data, num_s_features))
a_s = None
y_s = []
for i in range(5):
    y_s.extend([i] * 4)

y_s = torch.tensor(y_s, dtype = torch.long)

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
def test_accuracy(model, labels: torch.Tensor):
    
    with torch.no_grad():
        output = model(x_o)

    predicted = torch.argmax(output, dim = 1)
    predicted = predicted.numpy()
    labels = labels.numpy()

    acc = accuracy_score(labels, predicted)

    print("acc: ", acc)

    cm = confusion_matrix(labels, predicted)
    print(cm)

    
#%%
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr = 0.1)

for i in range(2000):
    optim.zero_grad()
    output = model(x_o)

    loss = loss_fn(output, y_o)

    if i % 200 == 0:
        print("loss: ", loss.item())
        test_accuracy(model, y_o)

    loss.backward()
    optim.step()


#%%
"""

"""
