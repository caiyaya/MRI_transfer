import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
# from models.ResNet import ResNet


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, modality=8, device = 'cpu'):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1,
                               kernel_size=modality,
                               dilation=int(channel//modality),
                               bias=False) #padding=(k_size - 1) // 2 # cross-channel interaction  EMCA
        self.conv2 = nn.Conv1d(1, 1,
                               kernel_size=channel//modality,
                               stride=channel//modality,
                               bias=False)#no cross-channel interaction
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.modality = modality
        self.device = device

    def forward(self, x):
        # feature descriptor on the global spatial information
        b, c, h, w = x.size()
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y1 = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y1 = torch.cat(tuple(y1 for _ in range(self.modality)),1)
        y2 = self.conv2(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = torch.cat(tuple(y2[:,_:_+1,:,:].expand(b,self.channel//self.modality,1,1) for _ in range(self.modality)),1)

        # Multi-scale information fusion
        v = self.sigmoid(y1+y2)
        v = v.expand(b,c,h,w)
        return x * v


class ECAResNet(nn.Module):
    def __init__(self, block, layers, in_channel=2, modality=2, out_channel=32, device = 'cpu'):
        super(ECAResNet, self).__init__()
        self.device = device
        self.modality = modality
        self.firstplanes = self.modality**2
        while self.firstplanes < 64:
            self.firstplanes *=2
        self.inplanes = self.firstplanes
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=self.firstplanes, kernel_size=3, stride=1, padding=1, groups=self.modality, bias=False)
        self.bn1 = nn.BatchNorm2d(self.firstplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layers = []
        for i, nblock in enumerate(layers):
            if i == 0:
                self.layers.append(self._make_layer(block, self.firstplanes * (2**i), nblock))
            else:
                self.layers.append(self._make_layer(block, self.firstplanes * (2**i), nblock, stride=2))
        self.layers = nn.Sequential(*self.layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.firstplanes * (2**i), out_channel)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride = 1):
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
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x= layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def get_embedding(self, x):
        return self.forward(x)


class ECABasicBlock(nn.Module):
    expansion: int = 1
    def __init__(self, inplanes, planes, stride=1, modality=8, device = 'cpu', downsample = None):#(self, inplanes, planes, stride=1, reduction=16)
        super(ECABasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,stride=stride,groups=modality, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,stride=1,groups=modality, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, modality, device)
        self.downsample = downsample
        self.stride = stride
        self.outdim = planes

    def forward(self, x):
        #print("block input:")
        #print(x.shape)
        residual = x
        #print("downsample out:")
        #print(residual.shape)
        out = self.conv1(x)
        #print("block conv1 out:")
        #print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #print("block conv2 out:")
        #print(out.shape)
        out = self.bn2(out)
        out = self.eca(out)
        #print("block se out:")
        #print(out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        #print("block out:")
        #print(out.shape)

        return out



def eca_resnet20(**kwargs):
    """Constructs a ResNet-20 model.

    """
    model = ECAResNet(ECABasicBlock, [3, 3, 3], **kwargs)
    return model



def eca_resnet0(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = models.ResNet(ECABasicBlock, [2], **kwargs)
    return model

# def eca_resnet18(**kwargs):
#     """Constructs a ResNet-18 model.
#
#     Args:
#         k_size: Adaptive selection of kernel size
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         num_classes:The classes of classification
#     """
#     model = ResNet(ECABasicBlock, [2, 2, 2, 2], **kwargs)
#     return model
#
#
# def eca_resnet34(k_size=[3, 3, 3, 3]):
#     """Constructs a ResNet-34 model.
#
#     Args:
#         k_size: Adaptive selection of kernel size
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         num_classes:The classes of classification
#     """
#     model = ResNet(ECABasicBlock, [3, 4, 6, 3], k_size=k_size)
#     return model
#
#
# def eca_resnet50(k_size=[3, 3, 3, 3]):
#     """Constructs a ResNet-50 model.
#
#     Args:
#         k_size: Adaptive selection of kernel size
#         num_classes:The classes of classification
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     print("Constructing eca_resnet50......")
#     model = ResNet(ECABottleneck, [3, 4, 6, 3], k_size=k_size)
#     return model
#
#
# def eca_resnet101(k_size=[3, 3, 3, 3]):
#     """Constructs a ResNet-101 model.
#
#     Args:
#         k_size: Adaptive selection of kernel size
#         num_classes:The classes of classification
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(ECABottleneck, [3, 4, 23, 3], k_size=k_size)
#     return model
#
#
# def eca_resnet152(k_size=[3, 3, 3, 3]):
#     """Constructs a ResNet-152 model.
#
#     Args:
#         k_size: Adaptive selection of kernel size
#         num_classes:The classes of classification
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(ECABottleneck, [3, 8, 36, 3], k_size=k_size)
#     return model