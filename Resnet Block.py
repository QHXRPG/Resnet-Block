import torch
import torch.nn as nn

class Resnet_Block(nn.Module):
    def __init__(self):
        super(Resnet_Block, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=64,kernel_size=1,stride=1,padding=0),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=64,out_channels=256,kernel_size=1,stride=1,padding=0))
        self.relu = nn.ReLU()

    def forward(self,x):
        return self.relu(self.block(x)+x)

resnet_block=Resnet_Block()
x=torch.rand(256,224,224)
c=resnet_block(x)