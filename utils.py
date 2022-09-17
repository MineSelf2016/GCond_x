#%%
import numpy as np
import torch

from collections import Counter
#%%
class Map(dict):

    def __init__(self):
        super.__init__(self)

    def __getattr__(self, name):
        try:
            return self[name]
        except:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value
