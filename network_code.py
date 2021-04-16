import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]

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
        self.conv2 = conv3x3x3(planes, planes, stride = 1)
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
    

#class BasicBlock2(nn.module):
    #code it in later    
    
#class BasicBlock3(nn.module):             
    #code it in later    
    
 
    

class SiameseNetwork3D(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()
        
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        
        self.in_planes = block_inplanes[0]
        
        self.no_max_pool = no_max_pool
        
        # Out=((Input_size − kernal_size + 2*Padding )/ Stride) + 1
        
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 3, 3),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # stride of block should be 2
        self.layer1 = self._make_layer(block, 
                                       block_inplanes[0], 
                                       layers[0],
                                       shortcut_type,
                                       stride = 2) 
        
        #block with stride 1
        self.layer2 = self._make_layer(block, 
                                       block_inplanes[1], 
                                       layers[1],
                                       shortcut_type,
                                       stride = 1)
        #Output before concat
        
        #stride of block shoud be 2
        self.layer3 = self._make_layer(block, 
                                       block_inplanes[2], 
                                       layers[2],
                                       shortcut_type,
                                       stride = 2)
        #stride of block should be 1
        self.layer4 = self._make_layer(block, 
                                       block_inplanes[2], 
                                       layers[3],
                                       shortcut_type,
                                       stride = 1)
        
        #stride of block shoud be 2
        self.layer5 = self._make_layer(block, 
                                       block_inplanes[3], 
                                       layers[4],
                                       shortcut_type,
                                       stride = 2)
        #stride of block should be 1
        self.layer6 = self._make_layer(block, 
                                       block_inplanes[3], 
                                       layers[5],
                                       shortcut_type,
                                       stride = 1)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        
        
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out
        
        
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))
                
                
            layers = []
            layers.append(
                block(in_planes=self.in_planes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample))
            self.in_planes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.in_planes, planes)) 
                
                
            return nn.Sequential(*layers)           
         
    def forward_one(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
                
        x = self.layer1(x)
        x = self.layer2(x)
                            
    def concat(self,input1, input2):
        # input 1 is be the 1st image and input 2 are all the consecutive images
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        # concatenation of all the data along the feature map dimension
        output_concat = torch.cat((output1,output2),0)
        return output_concat
        
    def forward_two(self,output_concat):
        x = self.layer3(output_concat)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avgpool(x)
        x = self.fc(x)
            
        return x


def generate_model():  
    return None  
        