import torch
import torch.nn as nn
import torch.nn.init as init
from models.ECAResNet import ECABasicBlock

class CifarECAResNet(nn.Module):
    def __init__(self, block, layers, k_size=[2, 2, 2], modality=2, in_channel = 1, device = 'cpu'):
        super(CifarECAResNet, self).__init__()
        self.device = device
        self.modality = modality
        self.firstplanes = modality *2
        self.inplanes = modality*2
        self.conv1 = nn.Conv2d(modality, self.inplanes, kernel_size=3, stride=1, padding=1, groups=modality, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layers = []
        for i,nblock in enumerate(layers):
            if i == 0:
                self.layers.append(self._make_layer(block, self.firstplanes * (2**i), nblock))
            else:
                self.layers.append(self._make_layer(block, self.firstplanes * (2**i), nblock, stride=2))
        self.layers = nn.Sequential(*self.layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride = 1):
        # strides = [stride] + [1] * (blocks - 1)#[1,1,1]
        # layers = []
        # for stride in strides:
        #     #print(stride)#2 1 1
        #     layers.append(block(self.inplane, planes, stride, k_size, self.modality, self.device))
        #     self.inplane = planes

        # return nn.Sequential(*layers)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kwargs = {}
        if not self.modality is None:
            kwargs['modality'] = self.modality
        layers.append(block(self.inplanes, planes, stride = stride, downsample = downsample, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, **kwargs))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        #print("input out:")
        #print(x.shape) #(40, 1, 32, 32)(40, 1, 128, 128)
        x = self.conv1(x)
        #print("conv1 out:")
        #print(x.shape) #(40, 16, 32, 32)(40, 16, 128, 128)

        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x= layer(x)

        x = self.avgpool(x)
        #print("avgpool out:")
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print("x.view out:")
        #print(x.shape)
        # x = self.fc(x)

        return x
    def get_embedding(self, x):
        return self.forward(x)

def eca_resnet20(**kwargs):
    """Constructs a ResNet-20 model.

    """
    model = CifarECAResNet(ECABasicBlock, [3, 3, 3], **kwargs)
    return model
