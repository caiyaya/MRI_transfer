import torch
import torch.nn as nn
import math
# import torch.utils.model_zoo as model_zoo
from .eca_module import eca_layer
import torch.nn.init as init


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ECABasicBlock(nn.Module):
    # 和论文图6(c)一致
    """
    Arg:
        inplanes: 输入的通道数
        planes: 输出的通道数
        downsample:是否进行下采样

    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)  # inplanes：输入通道数；planes：输出通道数
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ECABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes * 4, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def eca_resnet18(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet34(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet50(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing eca_resnet50......")
    model = ResNet(ECABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet101(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-101 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet152(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-152 model.

    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class CifarECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, k_size=3, modality=19):  # (self, inplanes, planes,
        # stride=1, reduction=16)
        super(CifarECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size, modality)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        elif inplanes == planes and inplanes == modality*2 and stride == 2:
            # print("gggggg")
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        # print("block input:")
        # print(x.shape)
        residual = self.downsample(x)
        # print("downsample out:")
        # print(residual.shape)
        out = self.conv1(x)
        # print("block conv1 out:")
        # print(out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # print("block conv2 out:")
        # print(out.shape)
        out = self.bn2(out)
        out = self.eca(out)
        # print("block se out:")
        # print(out.shape)

        out += residual
        out = self.relu(out)
        # print("block out:")
        # print(out.shape)
        return out


class CifarECAResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=128, k_size=[2, 2, 2], modality=19):
        super(CifarECAResNet, self).__init__()
        self.inplane = modality*2
        self.conv1 = nn.Conv2d(modality, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, modality*2, blocks=n_size, stride=2, k_size=int(k_size[0]), modality=modality)  # stride=1
        # self.layer2 = self._make_layer(block, modality*4, blocks=n_size, stride=2, k_size=int(k_size[1]), modality=modality)
        # self.layer3 = self._make_layer(block, modality*8, blocks=n_size, stride=2, k_size=int(k_size[2]), modality=modality)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(modality*2, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, k_size, modality):
        strides = [stride] + [1] * (blocks - 1)  # [1,1,1]
        layers = []
        for stride in strides:
            # print(stride)#2 1 1
            layers.append(block(self.inplane, planes, stride, k_size, modality))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # print("input out:")
        # print(x.shape) #(40, 1, 32, 32)(40, 1, 128, 128)
        x = self.conv1(x)
        # print("conv1 out:")
        # print(x.shape) #(40, 16, 32, 32)(40, 16, 128, 128)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # print("layer1 out:")
        # print(x.shape)
        # #x = self.layer2(x)
        # print("layer2 out:")
        # print(x.shape)
        # #x = self.layer3(x)
        # print("layer3 out:")
        # print(x.shape)

        x = self.avgpool(x)
        # print("avgpool out:")
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print("x.view out:")
        # print(x.shape)
        x = self.fc(x)

        return x

    def get_embedding(self, x):
        return self.forward(x)


def eca_resnet20(**kwargs):
    """Constructs a ResNet-20 model.

    """
    model = CifarECAResNet(CifarECABasicBlock, 2, **kwargs)
    return model


class DualembeddingNet(nn.Module):
    def __init__(self, embedding_net1,embedding_net2):
        super(DualembeddingNet, self).__init__()
        self.embedding_net1 = embedding_net1
        self.embedding_net2 = embedding_net2
        self.dp = nn.Dropout(0.5)

    def forward(self, x1, x2):
        output1 = self.embedding_net1(x1)
        output2 = self.embedding_net2(x2)
        totalembedding=torch.cat([output1,output2],dim=1)
        return totalembedding

    def get_embedding(self, x1, x2):
        return self.forward(x1,x2)