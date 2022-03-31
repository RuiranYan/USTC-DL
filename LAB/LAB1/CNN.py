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
        self.conv1 = nn.Conv2d(32,64,3,2,1) #(32,64)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64,128,3,2,1) #(16,128)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128,256,3,2,1) #(8,256)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.AvgPool2d(4,4,0) #(2,256)
        
        self.mlp1 = nn.Linear(2*2*256,512)
        self.mlp2 = nn.Linear(512, 200)

        ########################    END   ##########################
        self.dropout = nn.Dropout(0.15)
                
    def forward(self, input):
        x = F.relu(self.bn0(self.dropout(self.conv0(input))))

        ######################## task 1.2 ##########################
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.pool(x)
        
        x = F.relu(self.mlp1(x.view(x.size(0),-1)))
        
        x = self.mlp2(x)
        
        # Tips x = x.view(-1, 3*3*512)
        ########################    END   ##########################

        return x