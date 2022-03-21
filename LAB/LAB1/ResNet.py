import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ResNetLayer(nn.Module):
    def __init__(self, in_feature_maps, out_feature_maps, downsample = True):
        super(ResNetLayer, self).__init__()

        self.stride = 2 if downsample == True else 1
        self.conv0 = nn.Conv2d(in_feature_maps, out_feature_maps, 3, stride = self.stride, padding = 1)
        self.bn0 = nn.BatchNorm2d(out_feature_maps)
        self.conv1 = nn.Conv2d(out_feature_maps, out_feature_maps, 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_feature_maps)

        self.skipconn_cnn = nn.Conv2d(in_feature_maps, out_feature_maps, kernel_size=1, stride=self.stride, padding = 0)
        self.skipconn_bn = nn.BatchNorm2d(out_feature_maps)

    def forward(self, input):
        ######################## task 2.1 ##########################
        x = F.relu(self.bn0(self.conv0(input)))
        x = self.bn1(self.conv1(x))
        x = x + self.skipconn_bn(self.skipconn_cnn(input))
        return F.relu(x)
        ########################    END   ##########################


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        self.conv0 = nn.Conv2d(3, 64, 3, stride = 1, padding = 1) #(64*64*3) -> (64*64*64)
        self.bn0 = nn.BatchNorm2d(64)
        
        ######################## task 2.2 ##########################
        self.block1 = ResNetLayer(64,32)
        self.block2 = ResNetLayer(32,16)
        self.mlp1 = nn.Linear(16*16*16,512)
        self.mlp12 = nn.Linear(512,200)

        ########################    END   ##########################

        self.dropout = nn.Dropout(0.15)

    def forward(self, input):
        x = F.relu(self.bn0(self.dropout(self.conv0(input))))
        
        ######################## task 2.3 ##########################
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = F.relu(self.mlp1(x.view(x.size(0),-1)))
        x = self.mlp2(x)

        ########################    END   ##########################

        return x