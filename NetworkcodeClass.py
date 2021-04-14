import torch.nn as nn
import torch.nn.functional as F

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes,
                     out_planes, 
                     kernel_size=3,
                     stride=stride, 
                     padding=1, 
                     bias=False)
    
    
def conv1x1x1(in_planes, out_planes, stride=1):
    # 1x1x1 convolutions to be used during bottlenecks
    return nn.Conv3d(in_planes, 
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)  
    
    
class BasicBlock1(nn.Module):
    # From the GitRepo
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class BasicBlock2(nn.module):
    
    
class BasicBlock3(nn.module):             
    
    
 
    

class SiameseNetwork3D(nn.Module):
    def __init__(self):