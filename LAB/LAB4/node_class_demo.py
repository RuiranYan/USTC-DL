import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# Overview of Node Classification with GNN
# ----------------------------------------
#
# One of the most popular and widely adopted tasks on graph data is node
# classification, where a model needs to predict the ground truth category
# of each node. Before graph neural networks, many proposed methods are
# using either connectivity alone (such as DeepWalk or node2vec), or simple
# combinations of connectivity and the node's own features.  GNNs, by
# contrast, offers an opportunity to obtain node representations by
# combining the connectivity and features of a *local neighborhood*.
#
# `Kipf et
# al., <https://arxiv.org/abs/1609.02907>`__ is an example that formulates
# the node classification problem as a semi-supervised node classification
# task. With the help of only a small portion of labeled nodes, a graph
# neural network (GNN) can accurately predict the node category of the
# others.
# 
# This tutorial will show how to build such a GNN for semi-supervised node
# classification with only a small number of labels on the Cora
# dataset,
# a citation network with papers as nodes and citations as edges. The task
# is to predict the category of a given paper. Each paper node contains a
# word count vector as its features, normalized so that they sum up to one,
# as described in Section 5.2 of
# `the paper <https://arxiv.org/abs/1609.02907>`__.
# 
# Loading Cora Dataset
# --------------------
# 

import dgl.data
from load_graph import Load_graph, simple_dataloader
#dataset = dgl.data.CoraGraphDataset()
dataset = Load_graph('cora', 'node')
print('Number of categories:', dataset.num_classes)
# loader = simple_dataloader(dataset=dataset)


######################################################################
# A DGL Dataset object may contain one or multiple graphs. The Cora
# dataset used in this tutorial only consists of one single graph.
# 

print ('dataset len:',len(dataset))
# for g in loader:
#     print(g)
g = dataset[0]
from dgl import DropEdge
# g = dgl.add_self_loop(g)
# g = dgl.remove_self_loop(g)



######################################################################
# A DGL graph can store node features and edge features in two
# dictionary-like attributes called ``ndata`` and ``edata``.
# In the DGL Cora dataset, the graph contains the following node features:
# 
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
# 

print('Node features')
print(g.ndata)
print('Edge features')
print(g.edata)


######################################################################
# Defining a Graph Convolutional Network (GCN)
# --------------------------------------------
# 
# This tutorial will build a two-layer `Graph Convolutional Network
# (GCN) <http://tkipf.github.io/graph-convolutional-networks/>`__. Each
# layer computes new node representations by aggregating neighbor
# information.
# 
# To build a multi-layer GCN you can simply stack ``dgl.nn.GraphConv``
# modules, which inherit ``torch.nn.Module``.
#


class PairNorm(nn.Module):
    def __init__(self, mode='PN', scale=1):
        """
            mode:
              'None' : No normalization
              'PN'   : Original version
              'PN-SI'  : Scale-Individually version
              'PN-SCS' : Scale-and-Center-Simultaneously version

            ('SCS'-mode is not in the paper but we found it works well in practice,
              especially for GCN and GAT.)
            PairNorm is typically used after each graph convolution operation.
        """
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


from dgl.nn import GraphConv



class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)
        # self.conv3 = GraphConv(h_feats, h_feats, allow_zero_in_degree=True)
        # self.dropout = DropEdge(p=0.001)
        self.norm = PairNorm()
    
    def forward(self, g, in_feat):
        # g = self.dropout(g)
        h = self.conv1(g, in_feat)
        # h = F.relu(h)
        # h = self.conv3(g, h)
        # h = self.norm(h)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
# Create the model with given dimensions
model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)


######################################################################
# DGL provides implementation of many popular neighbor aggregation
# modules. You can easily invoke them with one line of code.
# 


######################################################################
# Training the GCN
# ----------------
# 
# Training this GCN is similar to training other PyTorch neural networks.
# 

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(100):
        # model.train()
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()

        logits = model(g, features)
        pred = logits.argmax(1)
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (e+1) % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                e+1, loss, val_acc, best_val_acc, test_acc, best_test_acc))

    model.eval()
    logits = model(g, features)
    pred = logits.argmax(1)
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    print(test_acc)


model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)



######################################################################
# Training on GPU
# ---------------
# 
# Training on GPU requires to put both the model and the graph onto GPU
# with the ``to`` method, similar to what you will do in PyTorch.
# 
# .. code:: python
#
#    g = g.to('cuda')
#    model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to('cuda')
#    train(g, model)
#

