import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self,input_shape=(7,14,14),num_res_blocks=5,num_hidden=64,num_channels_conv=32):
        super().__init__()
        self.startBlock=nn.Sequential(
            nn.Conv2d(input_shape[0], num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backbone=nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_res_blocks)]
        )
        self.policyHead=nn.Sequential(
            nn.Conv2d(num_hidden, num_channels_conv, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(num_channels_conv*input_shape[1]*input_shape[2],input_shape[0]*input_shape[1]*input_shape[2]),
            nn.Softmax(dim=-1)
        )
        self.valueHead=nn.Sequential(
            nn.Conv2d(num_hidden,input_shape[0],kernel_size=3,padding=1),
            nn.BatchNorm2d(input_shape[0]),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(input_shape[0]*input_shape[1]*input_shape[2],1),
            nn.Tanh()
        )
    def forward(self,x):
        x=self.startBlock(x)
        for block in self.backbone:
            x=block(x)
        p=self.policyHead(x)
        v=self.valueHead(x)
        return p,v


class ResBlock(nn.Module):
    def __init__(self,num_hidden):
        super().__init__()
        self.conv1=nn.Conv2d(num_hidden,num_hidden,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(num_hidden)
        self.cnv2=nn.Conv2d(num_hidden,num_hidden,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(num_hidden)
    def forward(self,x):
        residual=x
        x=F.relu(self.bn1(self.conv1(x)))
        x=self.bn2(self.cnv2(x))
        x+=residual
        x=F.relu(x)
        return x