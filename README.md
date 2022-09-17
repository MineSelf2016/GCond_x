## The Initial Code of Graph Condensation


How to encapusalate the dict class.



1. Construct the features;

How to initialize a network? To make it generated from a specific distribution.

**If we use the symmetric form of the adjacent matrix, it will be definitely the same in the first forward!**


2. Construct the adjacent matrix;


My implementation

```python
    a_l = x.repeat(1, num_nodes).reshape(num_nodes, num_nodes, num_features)

    a_r = x.repeat(num_nodes, 1).reshape(num_nodes, num_nodes, num_features)

    a_syn = torch.cat([a_l, a_r], dim = 2)
    

```


Wei's implem





3. Construct the labels;




