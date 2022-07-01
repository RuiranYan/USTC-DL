import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from load_graph import Load_graph, simple_dataloader

from dgl import DropEdge
dataset = Load_graph('ppi', 'node')
print ('dataset len:',len(dataset))
print(dataset)
g = dataset[0]
print(g.ndata)

# load data
trainloader = simple_dataloader(dataset=dataset[:18])
valloader = simple_dataloader(dataset=dataset[18:19])
testloader = simple_dataloader(dataset=dataset[19:20])


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
        self.dropout = DropEdge(p=0.001)
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

num_classes = g.ndata['label'].shape[1]
model = GCN(g.ndata['feat'].shape[1], 16, num_classes)

def train(trainloader,valloader,testloader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    for e in range(100):
        model.train()
        train_acc_list = []
        for g in trainloader:

            features = g.ndata['feat']
            labels = g.ndata['label']

            logits = model(g, features)

            train_loss = F.binary_cross_entropy_with_logits(logits,labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        for val in valloader:
            features = val.ndata['feat']
            labels = val.ndata['label']
            logits = model(val, features)
            pred = F.sigmoid(logits) > 0.5
            val_acc = (pred==labels).float().mean()

        for test in testloader:
            features = test.ndata['feat']
            labels = test.ndata['label']
            logits = model(test, features)
            pred = F.sigmoid(logits) > 0.5
            test_acc = (pred == labels).float().mean()

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    print("best test acc:",best_test_acc)



train(trainloader,valloader,testloader, model)