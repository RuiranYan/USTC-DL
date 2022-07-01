# GCN-demo

We will leverage DGL to finish the experiment. [What's DGL?](https://docs.dgl.ai/en/latest/index.html)

Basic Knowledge about DGL:
- [Graph](https://docs.dgl.ai/en/latest/guide/graph.html)
- [Message Passing](https://docs.dgl.ai/en/latest/guide/message.html)
- [Building GNN Modules](https://docs.dgl.ai/en/latest/guide/nn.html)


## Environment

* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.5

* [DGL](https://www.dgl.ai/pages/start.html) >= 0.8

## Dataset
You can load dataset from [load_graph.py](./load_graph.py)
```python3
$ from load_graph import load_graph
$ dataset = Load_graph('cora')

# Access the first graph, it will return a DGLGraph
# For cora, it only consists of one single graph 
$ g = dataset[0]
$ print(g)
Graph(num_nodes=2708, num_edges=10556,
      ndata_schemes={'feat': Scheme(shape=(1433,), dtype=torch.float32), 'label': Scheme(shape=(), dtype=torch.int64), 'test_mask': Scheme(shape=(), dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'train_mask': Scheme(shape=(), dtype=torch.bool)}
      edata_schemes={'__orig__': Scheme(shape=(), dtype=torch.int64)})

# Access graph node features
$ print(g.ndata)
{'feat': tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0526, 0.0000]]), 'label': tensor([4, 4, 4,  ..., 4, 3, 3]), 'test_mask': tensor([ True,  True, False,  ..., False, False, False]), 'val_mask': tensor([False, False,  True,  ..., False, False, False]), 'train_mask': tensor([False, False, False,  ..., False, False, False])}
# - ``train_mask``: A boolean tensor indicating whether the node is in the
#   training set.
#
# - ``val_mask``: A boolean tensor indicating whether the node is in the
#   validation set.
#
# - ``test_mask``: A boolean tensor indicating whether the node is in the
#   test set.
#
# - ``label``: The ground truth node category.
#
# -  ``feat``: The node features.
```

**Attention** : 
- `train_mask`, `val_mask`, `test_mask`, `label` are just for node classification. For edge prediction, you should carefully study the case [link_pred_demo.py](./link_pred_demo.py)

- PPI dataset consists of 20 graph and others consists of single graph. For PPI, you should train its graph one by one. I provide a [simple_dataloader](./load_graph.py#L4) to help you.
    ```python3
    from load_graph import simple_dataloader, load_graph
    # dataset can be list, tuple or other object support __getitem__
    ## For node classification, it will help for ppi
    dataset = Load_graph('ppi')
    loader = simple_dataloader(dataset=dataset)
    for g in loader:
        print(g)

    ## For edge prediction, it will help
    data = [[g1, pos_g1, neg_g1], [g2, pos_g2, neg_g2], ...]
    loader = simple_dataloader(dataset=data)
    for g, pos_g, neg_g in loader:
        print(g, pos_g, neg_g)
    ```

- For node classification, the labels for cora and citeseer are integer and you can access `dataset.num_classes` to get the number of categories. However, for ppi, it's label is vector and you should consider which loss function works. 

## Case Study
- Link Prediction
    - [Link Prediction using Graph Neural Networks](https://docs.dgl.ai/en/latest/tutorials/blitz/4_link_predict.html#sphx-glr-download-tutorials-blitz-4-link-predict-py)
    - [code](./link_pred_demo.py)
    - ```python3 link_pred_demo.py```
- Node Classification
    - [Node Classification with DGL](https://docs.dgl.ai/en/latest/tutorials/blitz/1_introduction.html)
    - [code](./node_class_demo.py)
    - ```python3 node_class_demo.py```

## Tasks
Given three datasets (cora, citeseer, ppi), implement the GCN algorithm for node classification and link prediction, and analyse the effect of self-loop, number of layers, DropEdge, PairNorm, activation function and other factors on performance

## Tips
- self-loop : `dgl.remove_self_loop`, `dgl.add_self_loop`
- DropEdge : `dgl.transforms.DropEdge`
- Your best friends : [google](https://www.google.com), [baidu](https://www.baidu.com), [bing](https://www.bing.com)
- Other GNN framework is ok, however, you have to prepare everything yourself
- 为什么是英文呢，因为本次实验是npy写的框架，大家521快乐！
- 截止时间: 2022/6/22 23:59:59
