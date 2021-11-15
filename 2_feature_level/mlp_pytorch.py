"""MLP model is used to classify the fusioned Raster and point cloud

Xiaoqiang Liu      2019/07/25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import numpy as np
 
class MLP(nn.Module):
    # MLP  network
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)
    def __init__(self, in_channels=192, out_channels=8):
        super(MLP, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels,128,1)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv1d(128,out_channels,1)

        self.apply(self.weight_init)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.log_softmax(self.conv2(x), dim=1)
        return x

if __name__ == "__main__":
    point_cloud = np.random.rand(2,192,8192)
    point_cloud = torch.from_numpy(point_cloud)
    point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
    net = MLP()
    net.eval()
    out = net(point_cloud)
    print(out.shape)
