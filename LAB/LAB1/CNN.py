import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv0 = nn.Conv2d(3, 32, 3, stride = 1, padding = 1) #(64*64*3) -> (64*64*32)
        self.bn0 = nn.BatchNorm2d(32)
        
        ######################## task 1.1 ##########################
        self.conv1 = nn.Conv2d(32,64,3,3,1) #(64*64*32)->(22*22*64)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(64,16,2,2,0) #(22*22*128)->(11*11*16)
        self.bn2 = nn.BatchNorm2d(32)
        self.mlp1 = nn.Linear(11*11*16,512)
        self.mlp2 = nn.Linear(512, 200)

        ########################    END   ##########################
        self.dropout = nn.Dropout(0.15)
                
    def forward(self, input):
        x = F.relu(self.bn0(self.dropout(self.conv0(input))))

        ######################## task 1.2 ##########################
        x = F.relu(self.bn1(self.dropout(self.conv1(x))))
        x = F.relu(self.bn2(self.dropout(self.conv2(x))))
        x = F.relu(self.mlp1(x.view(x.size(0),-1)))
        x = self.mlp2(x)
        
        # Tips x = x.view(-1, 3*3*512)
        ########################    END   ##########################

        return x