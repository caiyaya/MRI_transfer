import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.autograd import Function
# from model.se_module import CifarSEBasicBlock, SEBasicBlock_IBN, SEBasicBlock_IBN_cnsn
# import model.usps as usps
# import model.mice2human as mice2human
from .eca_module import *
# from .se_module import *
# from .cnsn import CrossNorm, SelfNorm, CNSN


# =================================== Basic layers ===================================
class Conv_Layer(nn.Module):
    """Convolution layer (conv + bn + relu)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(Conv_Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1(nn.Module):
    """1x1 convolution + bn + relu."""
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1Linear(nn.Module):
    """1x1 convolution + bn (w/o non-linearity)."""
    def __init__(self, in_channels, out_channels, stride=1, bn=True):
        super(Conv1x1Linear, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False
        )
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


# =================================== Omni-Scale SE Bottleneck ===================================
class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, T=3):
        super(OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T+1):
            self.conv2 += [SEBlockStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class SEBlockStream(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(SEBlockStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(depth)
        layers = []
        layers += [CifarSEBasicBlock(in_channels, out_channels)]
        for i in range(depth - 1):
            layers += [CifarSEBasicBlock(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ChannelGate(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""
    def __init__(
        self,
        in_channels,
        num_gates=None,
        return_gates=False,
        gate_activation='sigmoid',
        reduction=16,
        layer_norm=False
    ):
        super(ChannelGate, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            in_channels,
            in_channels // reduction,
            kernel_size=1,
            bias=True,
            padding=0
        )
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(
            in_channels // reduction,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU()
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x

# =================================== feature extract module OSSE ===================================
class Feature_Extract_Net_OSSE(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_OSSE, self).__init__()
        self.in_channels = 3 # 如果选择序列order有改变，要修改哦
        # self.out_channels = [16, 64, 96, 128]
        self.out_channels = [16, 64, 96, 128]
        self.blocks = [[OSBlock, OSBlock], [OSBlock, OSBlock], [OSBlock, OSBlock]]

        self.conv = Conv_Layer(self.in_channels, self.out_channels[0], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.osse1 = self.make_layer(self.blocks[0], self.out_channels[0], self.out_channels[1])
        self.tran1 = nn.Sequential(Conv1x1(self.out_channels[1], self.out_channels[1]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse2 = self.make_layer(self.blocks[1], self.out_channels[1], self.out_channels[2])
        self.tran2 = nn.Sequential(Conv1x1(self.out_channels[2], self.out_channels[2]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse3 = self.make_layer(self.blocks[2], self.out_channels[2], self.out_channels[3])
        self.tran3 = Conv1x1(self.out_channels[3], self.out_channels[3])


        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self.construct_fc_layer(self.out_channels[3], self.out_channels[3], dropout_p=None)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_layer(self, blocks, in_channels, out_channels):
        layers = []
        layers += [blocks[0](in_channels, out_channels)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels)]

        return nn.Sequential(*layers)

    def construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv(x)
        x0 = self.pool(x0)
        x1 = self.osse1(x0)
        x1 = self.tran1(x1)
        x2 = self.osse2(x1)
        x2 = self.tran2(x2)
        x3 = self.osse3(x2)
        x3 = self.tran3(x3)

        x3 = self.global_avgpool(x3)
        x3 = x3.view(x3.size(0), -1)
        x_last = self.fc(x3)

        return x1, x2, x3, x_last

# =================================== feature extract module SE ===================================


class Feature_Extract_Net_SE(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_SE, self).__init__()
        self.n_size = 3
        self.reduction = 16
        self.channel = 3  # 如果选择序列order有改变，要修改哦
        self.inplane = 16
        self.block = CifarSEBasicBlock

        self.conv1 = nn.Conv2d(self.channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, self.inplane, blocks=self.n_size, stride=1, reduction=self.reduction)
        self.layer2 = self._make_layer(self.block, self.inplane * 2, blocks=self.n_size, stride=2,
                                       reduction=self.reduction)
        self.layer3 = self._make_layer(self.block, self.inplane * 4, blocks=self.n_size, stride=2,
                                       reduction=self.reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_last = self.avgpool(x3)

        x_last = x_last.view(x_last.size(0), -1)

        return x1, x2, x3, x_last

# =================================== feature extract module IBN ===================================


class Feature_Extract_Net_IBN(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_IBN, self).__init__()
        self.n_size = 3
        self.reduction = 16
        self.channel = 3  # 如果选择序列order有改变，要修改哦
        self.inplane = 16
        self.block = SEBasicBlock_IBN

        self.conv1 = nn.Conv2d(self.channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, self.inplane, blocks=self.n_size, stride=1, reduction=self.reduction)
        self.layer2 = self._make_layer(self.block, self.inplane * 2, blocks=self.n_size, stride=2,
                                       reduction=self.reduction)
        self.layer3 = self._make_layer(self.block, self.inplane * 4, blocks=self.n_size, stride=2,
                                       reduction=self.reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_last = self.avgpool(x3)

        x_last = x_last.view(x_last.size(0), -1)

        return x1, x2, x3, x_last

# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#
#
# class Cifar_SEBasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, stride=1, reduction=16):
#         super(Cifar_SEBasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.se = SE_Layer(planes, reduction)
#         if inplanes != planes:
#             self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
#                                             nn.BatchNorm2d(planes))
#         else:
#             self.downsample = lambda x: x
#         self.stride = stride
#
#     def forward(self, x):
#         residual = self.downsample(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.se(out)
#
#         out = out + residual
#         out = self.relu(out)
#
#         return out

# =================================== Omni-Scale ECA Bottleneck ===================================
class ECAOSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, T=3):
        super(ECAOSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T+1):
            self.conv2 += [ECABlockStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)


class ECABlockStream(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(ECABlockStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(depth)
        layers = []
        layers += [ECABasicBlock(in_channels, out_channels)]
        for i in range(depth - 1):
            layers += [ECABasicBlock(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# =================================== feature extract module ECA ===================================


class Feature_Extract_Net_ECA(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_ECA, self).__init__()
        self.n_size = 3
        self.channel = 3  # 如果选择序列order有改变，要修改哦
        self.inplane = 16
        self.block = ECABasicBlock

        self.conv1 = nn.Conv2d(self.channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, self.inplane, blocks=self.n_size, stride=1)
        self.layer2 = self._make_layer(self.block, self.inplane * 2, blocks=self.n_size, stride=2)
        self.layer3 = self._make_layer(self.block, self.inplane * 4, blocks=self.n_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_last = self.avgpool(x3)

        x_last = x_last.view(x_last.size(0), -1)

        return x1, x2, x3, x_last

# =================================== feature extract module OSECA ===================================
class Feature_Extract_Net_OSECA(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_OSECA, self).__init__()
        self.in_channels = 3 # 如果选择序列order有改变，要修改哦
        # self.out_channels = [16, 64, 96, 128]
        self.out_channels = [16, 64, 96, 128]
        self.blocks = [[ECAOSBlock, ECAOSBlock], [ECAOSBlock, ECAOSBlock], [ECAOSBlock, ECAOSBlock]]

        self.conv = Conv_Layer(self.in_channels, self.out_channels[0], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.osse1 = self.make_layer(self.blocks[0], self.out_channels[0], self.out_channels[1])
        self.tran1 = nn.Sequential(Conv1x1(self.out_channels[1], self.out_channels[1]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse2 = self.make_layer(self.blocks[1], self.out_channels[1], self.out_channels[2])
        self.tran2 = nn.Sequential(Conv1x1(self.out_channels[2], self.out_channels[2]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse3 = self.make_layer(self.blocks[2], self.out_channels[2], self.out_channels[3])
        self.tran3 = Conv1x1(self.out_channels[3], self.out_channels[3])


        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self.construct_fc_layer(self.out_channels[3], self.out_channels[3], dropout_p=None)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_layer(self, blocks, in_channels, out_channels):
        layers = []
        layers += [blocks[0](in_channels, out_channels)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels)]

        return nn.Sequential(*layers)

    def construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv(x)
        x0 = self.pool(x0)
        x1 = self.osse1(x0)
        x1 = self.tran1(x1)
        x2 = self.osse2(x1)
        x2 = self.tran2(x2)
        x3 = self.osse3(x2)
        x3 = self.tran3(x3)

        x3 = self.global_avgpool(x3)
        x3 = x3.view(x3.size(0), -1)
        x_last = self.fc(x3)

        return x1, x2, x3, x_last

# =================================== feature extract module 普普通通的特征提取器 ===================================
def Generator(source, target):
    if (source == 'mice') & (target == 'human'):
        return mice2human.Feature()
    elif (source == 'human') & (target == 'mice'):
        return mice2human.Feature()

# =================================== domain_adversarial_Net ===================================
# todo layer 层是否需要加参数进行调整
class Domain_Adversarial_Net(nn.Module):
    def __init__(self):
        super(Domain_Adversarial_Net, self).__init__()
        self.fc_layer1 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=False))
        self.fc_layer2 = nn.Sequential(nn.Linear(64, 16), nn.ReLU(inplace=False))
        self.fc_layer3 = nn.Sequential(nn.Linear(16, 2))
        self.apply(init_weights)
        # self.fc_layer2 = nn.Sequential(nn.Linear(3072, 2048), nn.ReLU(inplace=False))
        # self.fc_layer3 = nn.Sequential(nn.Linear(2048, 2))

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)

    def forward(self, x):
        grl = GradientReversal()
        x = grl(x)
        # x = grad_reverse(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)

        return x

class Text_Domain_Adversarial_Net(nn.Module):
    def __init__(self):
        super(Text_Domain_Adversarial_Net, self).__init__()
        self.fc_layer1 = nn.Sequential(nn.Linear(30000, 96), nn.ReLU(inplace=False))
        self.fc_layer2 = nn.Sequential(nn.Linear(96, 32), nn.ReLU(inplace=False))
        self.fc_layer3 = nn.Sequential(nn.Linear(32, 2))
        self.apply(init_weights)

    def forward(self, x):
        grl = GradientReversal()
        x = grl(x)
        # x = grad_reverse(x)
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)

        return x

# 第一个版本
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_ = 1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# 第二个版本
class GradReverseFunction(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd=1.0):
    return GradReverseFunction(lambd)(x)

# =================================== source_domain_classifier 源域分类器===================================
class Label_Classifier(nn.Module):
    def __init__(self, inplane, class_num):
        super(Label_Classifier, self).__init__()
        self.fc_layer1 = nn.Sequential(nn.Linear(inplane, 64), nn.ReLU(inplace=False))
        self.fc_layer2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(inplace=False))
        self.fc_layer3 = nn.Sequential(nn.Linear(32, class_num))
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
    # 324
    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.fc_layer2(x)
        x = self.fc_layer3(x)
        return x


# =================================== 参数分类器===================================
class FeatureFc(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeatureFc, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).double()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size).double()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


# =================================== Long CDAN ===================================
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class CDAN_AdversarialNetwork(nn.Module):
  def __init__(self):
    super(CDAN_AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(128, 64)
    self.ad_layer2 = nn.Linear(64, 16)
    self.ad_layer3 = nn.Linear(16, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]
        self.random_matrix = [val.cuda() for val in self.random_matrix]


# =================================== text feature extract module ===================================
class Text_Fully_Connected_Net(nn.Module):
    def __init__(self, class_num):
        super(Text_Fully_Connected_Net, self).__init__()
        self.fc_layer1 = nn.Linear(30000, 96)
        self.fc_layer2 = nn.Linear(96, 64)
        self.fc_layer3 = nn.Linear(64, 32)
        self.fc_layer4 = nn.Linear(32, class_num)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, x):
        x = self.fc_layer1(x)
        x = self.relu1(x)
        x1 = self.dropout1(x)

        x1 = self.fc_layer2(x1)
        x1 = self.relu2(x1)
        x2 = self.dropout2(x1)

        x2 = self.fc_layer3(x2)
        x2 = self.relu3(x2)
        x3 = self.dropout3(x2)

        x3 = self.fc_layer4(x3)
        x4 = self.sigmoid(x3)
        return x1, x2, x3, x4

# =================================== feature extract module ECA_IBN ===================================

class Feature_Extract_Net_ECA_IBN(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_ECA_IBN, self).__init__()
        self.n_size = 3
        self.channel = 3  # 如果选择序列order有改变，要修改哦
        self.inplane = 16
        self.block = ECABasicBlock_IBN

        self.conv1 = nn.Conv2d(self.channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.block, self.inplane, blocks=self.n_size, stride=1)
        self.layer2 = self._make_layer(self.block, self.inplane * 2, blocks=self.n_size, stride=2)
        self.layer3 = self._make_layer(self.block, self.inplane * 4, blocks=self.n_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_last = self.avgpool(x3)

        x_last = x_last.view(x_last.size(0), -1)

        return x1, x2, x3, x_last
        
# =================================== feature extract module IBN_cnsn ===================================
class Feature_Extract_Net_IBN_cnsn(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_IBN_cnsn, self).__init__()
        self.n_size = 3
        self.reduction = 16
        self.channel = 3  # 如果选择序列order有改变，要修改哦
        self.inplane = 16
        self.block = SEBasicBlock_IBN_cnsn
        self.active_num = 1
        self.pos = 'post'
        self.beta = None
        self.crop = 'neither'
        self.cnsn_type = 'cn'
        print('**********************************************')
        print('ResNet34 with ibn, selfnorm and crossnorm...')

        self.conv1 = nn.Conv2d(self.channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        
        if self.beta is not None:
            print('beta: {}'.format(self.beta))

        if self.crop is not None:
            print('crop mode: {}'.format(self.crop))
        self.layer1 = self._make_layer(self.block, self.inplane, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        self.layer2 = self._make_layer(self.block, self.inplane * 2, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=2, ibn='a',
                                       reduction=self.reduction)
        self.layer3 = self._make_layer(self.block, self.inplane * 4, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=2, ibn='a',
                                       reduction=self.reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cn_modules = []
        self.initialize()
        if self.cnsn_type is not None and 'cn' in self.cnsn_type:
            # self.active_num = active_num
            assert self.active_num > 0
            print('active_num: {}'.format(self.active_num))
            self.cn_num = len(self.cn_modules)
            assert self.cn_num > 0
            print('cn_num: {}'.format(self.cn_num))

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
           # elif isinstance(m, nn.BatchNorm2d):
           #    init.constant_(m.weight, 1)
           #    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, CrossNorm):
                self.cn_modules.append(m)
                
    def _enable_cross_norm(self):
        active_cn_idxs = np.random.choice(self.cn_num, self.active_num, replace=False).tolist()
        assert len(set(active_cn_idxs)) == self.active_num
        # print('active_cn_idxs: {}'.format(active_cn_idxs))
        for idx in active_cn_idxs:
            self.cn_modules[idx].active = True
            
    def _make_layer(self, block, planes, blocks, pos, beta, crop, cnsn_type, stride, ibn, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, pos, beta, crop, cnsn_type, stride, ibn, reduction))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_last = self.avgpool(x3)

        x_last = x_last.view(x_last.size(0), -1)

        return x1, x2, x3, x_last


# =================================== feature extract module eca_IBN_cnsn ===================================
class Feature_Extract_Net_eca_IBN_cnsn(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_eca_IBN_cnsn, self).__init__()
        self.n_size = 3
        self.reduction = 16
        self.channel = 3  # 如果选择序列order有改变，要修改哦
        self.inplane = 16
        self.block = ECABasicBlock_IBN_cnsn
        self.active_num = 1
        self.pos = 'post'
        self.beta = None
        self.crop = 'neither'
        self.cnsn_type = 'cn'
        print('**********************************************')
        print('ResNet34 with ibn, selfnorm and crossnorm...')

        self.conv1 = nn.Conv2d(self.channel, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        
        if self.beta is not None:
            print('beta: {}'.format(self.beta))

        if self.crop is not None:
            print('crop mode: {}'.format(self.crop))
        self.layer1 = self._make_layer(self.block, self.inplane, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        self.layer2 = self._make_layer(self.block, self.inplane * 2, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=2, ibn='a',
                                       reduction=self.reduction)
        self.layer3 = self._make_layer(self.block, self.inplane * 4, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=2, ibn='a',
                                       reduction=self.reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cn_modules = []
        self.initialize()
        if self.cnsn_type is not None and 'cn' in self.cnsn_type:
            # self.active_num = active_num
            assert self.active_num > 0
            print('active_num: {}'.format(self.active_num))
            self.cn_num = len(self.cn_modules)
            assert self.cn_num > 0
            print('cn_num: {}'.format(self.cn_num))

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
           # elif isinstance(m, nn.BatchNorm2d):
           #    init.constant_(m.weight, 1)
           #    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, CrossNorm):
                self.cn_modules.append(m)
                
    def _enable_cross_norm(self):
        active_cn_idxs = np.random.choice(self.cn_num, self.active_num, replace=False).tolist()
        assert len(set(active_cn_idxs)) == self.active_num
        # print('active_cn_idxs: {}'.format(active_cn_idxs))
        for idx in active_cn_idxs:
            self.cn_modules[idx].active = True
            
    def _make_layer(self, block, planes, blocks, pos, beta, crop, cnsn_type, stride, ibn, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, pos, beta, crop, cnsn_type, stride, ibn, reduction))
            self.inplane = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_last = self.avgpool(x3)

        x_last = x_last.view(x_last.size(0), -1)

        return x1, x2, x3, x_last
     
# =================================== Omni-Scale ECAIBN Bottleneck ===================================
class ECAIBN_OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, T=3):
        super(ECAIBN_OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList()
        for t in range(1, T+1):
            self.conv2 += [ECAIBN_BlockStream(mid_channels, mid_channels, t)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)

class ECAIBN_BlockStream(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(ECAIBN_BlockStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(depth)
        layers = []
        layers += [ECABasicBlock_IBN(in_channels, out_channels)]
        for i in range(depth - 1):
            layers += [ECABasicBlock_IBN(out_channels, out_channels)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
        
# =================================== Omni-Scale IBN_cnsn Bottleneck ===================================
class SEIBN_cnsn_OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pos, beta, crop, cnsn_type, stride=1, ibn='a', reduction=4, T=3):
        super(SEIBN_cnsn_OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction
        
        self.expansion = 1
        if ibn == 'a':
            self.bn1 = IBN(mid_channels)
        else:
            self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        
        # self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.ModuleList()
        for t in range(1, T+1):
            self.conv2 += [SEIBN_cnsn_BlockStream(mid_channels, mid_channels, t, pos, beta, crop, cnsn_type)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
     
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
            
        if self.IN is not None and pos == 'post':
            self.cnsn = None
        else:
            assert cnsn_type in ['sn', 'cn', 'cnsn']

            if 'cn' in cnsn_type:
                print('using CrossNorm with crop: {}'.format(crop))
                crossnorm = CrossNorm(crop=crop, beta=beta)
            else:
                crossnorm = None

            if 'sn' in cnsn_type:
                print('using SelfNorm')
                if pos == 'pre':
                    selfnorm = SelfNorm(in_channels)
                else:
                    selfnorm = SelfNorm(out_channels * self.expansion)
            else:
                selfnorm = None

            self.cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

        self.pos = pos
        if pos is not None:
            print('{} in residual module: {}'.format(cnsn_type, pos))
            assert pos in ['residual', 'pre', 'post', 'identity']

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)

class SEIBN_cnsn_BlockStream(nn.Module):
    def __init__(self, in_channels, out_channels, depth, pos, beta, crop, cnsn_type):
        super(SEIBN_cnsn_BlockStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(depth)
        layers = []
        layers += [SEBasicBlock_IBN_cnsn(in_channels, out_channels, pos, beta, crop, cnsn_type)]
        for i in range(depth - 1):
            layers += [SEBasicBlock_IBN_cnsn(out_channels, out_channels, pos, beta, crop, cnsn_type)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)   
        
# =================================== Omni-Scale ECAIBN_cnsn Bottleneck ===================================
class ECAIBN_cnsn_OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pos, beta, crop, cnsn_type, stride=1, ibn='a', reduction=4, T=3):
        super(ECAIBN_cnsn_OSBlock, self).__init__()
        assert T >= 1
        assert out_channels >= reduction and out_channels % reduction == 0
        mid_channels = out_channels // reduction

        self.expansion = 1
        if ibn == 'a':
            self.bn1 = IBN(mid_channels)
        else:
            self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.IN = nn.InstanceNorm2d(out_channels, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        
        # self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.ModuleList()
        for t in range(1, T+1):
            self.conv2 += [ECAIBN_cnsn_BlockStream(mid_channels, mid_channels, t, pos, beta, crop, cnsn_type)]
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
     
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
            
        if self.IN is not None and pos == 'post':
            self.cnsn = None
        else:
            assert cnsn_type in ['sn', 'cn', 'cnsn']

            if 'cn' in cnsn_type:
                print('using CrossNorm with crop: {}'.format(crop))
                crossnorm = CrossNorm(crop=crop, beta=beta)
            else:
                crossnorm = None

            if 'sn' in cnsn_type:
                print('using SelfNorm')
                if pos == 'pre':
                    selfnorm = SelfNorm(in_channels)
                else:
                    selfnorm = SelfNorm(out_channels * self.expansion)
            else:
                selfnorm = None

            self.cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

        self.pos = pos
        if pos is not None:
            print('{} in residual module: {}'.format(cnsn_type, pos))
            assert pos in ['residual', 'pre', 'post', 'identity']

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = 0
        for conv2_t in self.conv2:
            x2_t = conv2_t(x1)
            x2 = x2 + self.gate(x2_t)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        return F.relu(out)

class ECAIBN_cnsn_BlockStream(nn.Module):
    def __init__(self, in_channels, out_channels, depth, pos, beta, crop, cnsn_type):
        super(ECAIBN_cnsn_BlockStream, self).__init__()
        assert depth >= 1, 'depth must be equal to or larger than 1, but got {}'.format(depth)
        layers = []
        layers += [ECABasicBlock_IBN_cnsn(in_channels, out_channels, pos, beta, crop, cnsn_type)]
        for i in range(depth - 1):
            layers += [ECABasicBlock_IBN_cnsn(out_channels, out_channels, pos, beta, crop, cnsn_type)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
        
# =================================== feature extract module OSECA_IBN ===================================
class Feature_Extract_Net_OSECA_IBN(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_OSECA_IBN, self).__init__()
        self.in_channels = 3 # 如果选择序列order有改变，要修改哦
        # self.out_channels = [16, 64, 96, 128]
        self.out_channels = [16, 64, 96, 128]
        self.blocks = [[ECAIBN_OSBlock, ECAIBN_OSBlock], [ECAIBN_OSBlock, ECAIBN_OSBlock], [ECAIBN_OSBlock, ECAIBN_OSBlock]]

        self.conv = Conv_Layer(self.in_channels, self.out_channels[0], kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.osse1 = self.make_layer(self.blocks[0], self.out_channels[0], self.out_channels[1])
        self.tran1 = nn.Sequential(Conv1x1(self.out_channels[1], self.out_channels[1]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse2 = self.make_layer(self.blocks[1], self.out_channels[1], self.out_channels[2])
        self.tran2 = nn.Sequential(Conv1x1(self.out_channels[2], self.out_channels[2]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse3 = self.make_layer(self.blocks[2], self.out_channels[2], self.out_channels[3])
        self.tran3 = Conv1x1(self.out_channels[3], self.out_channels[3])


        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self.construct_fc_layer(self.out_channels[3], self.out_channels[3], dropout_p=None)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def make_layer(self, blocks, in_channels, out_channels):
        layers = []
        layers += [blocks[0](in_channels, out_channels)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels)]

        return nn.Sequential(*layers)

    def construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv(x)
        x0 = self.pool(x0)
        x1 = self.osse1(x0)
        x1 = self.tran1(x1)
        x2 = self.osse2(x1)
        x2 = self.tran2(x2)
        x3 = self.osse3(x2)
        x3 = self.tran3(x3)

        x3 = self.global_avgpool(x3)
        x3 = x3.view(x3.size(0), -1)
        x_last = self.fc(x3)

        return x1, x2, x3, x_last
        
# =================================== feature extract module osIBN_cnsn ===================================
class Feature_Extract_Net_OSIBN_cnsn(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_OSIBN_cnsn, self).__init__()
        self.n_size = 3
        self.reduction = 4
        # self.channel = 3  # 如果选择序列order有改变，要修改哦

        self.in_channels = 3 # 如果选择序列order有改变，要修改哦
        self.out_channels = [16, 64, 96, 128]
        # self.inplane = 16
        # self.block = SEIBN_cnsn_OSBlock
        self.blocks = [[SEIBN_cnsn_OSBlock, SEIBN_cnsn_OSBlock], [SEIBN_cnsn_OSBlock, SEIBN_cnsn_OSBlock], [SEIBN_cnsn_OSBlock, SEIBN_cnsn_OSBlock]]
        
        self.active_num = 1
        self.pos = 'post'
        self.beta = None
        self.crop = 'neither'
        self.cnsn_type = 'cn'
        print('**********************************************')
        print('ResNet34 with osse, ibn, cnsn...')

        self.conv = nn.Conv2d(self.in_channels, self.out_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels[0])
        self.relu = nn.ReLU(inplace=True)
        
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.osse1 = self.make_layer(self.blocks[0], self.out_channels[0], self.out_channels[1], self.pos, self.beta, self.crop, self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        self.tran1 = nn.Sequential(Conv1x1(self.out_channels[1], self.out_channels[1]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse2 = self.make_layer(self.blocks[1], self.out_channels[1], self.out_channels[2], self.pos, self.beta, self.crop, self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        self.tran2 = nn.Sequential(Conv1x1(self.out_channels[2], self.out_channels[2]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse3 = self.make_layer(self.blocks[2], self.out_channels[2], self.out_channels[3], self.pos, self.beta, self.crop, self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        self.tran3 = Conv1x1(self.out_channels[3], self.out_channels[3])
        
        if self.beta is not None:
            print('beta: {}'.format(self.beta))

        if self.crop is not None:
            print('crop mode: {}'.format(self.crop))
        # self.layer1 = self._make_layer(self.block, self.inplane, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        # self.layer2 = self._make_layer(self.block, self.inplane * 2, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=2, ibn='a',
        #                               reduction=self.reduction)
        # self.layer3 = self._make_layer(self.block, self.inplane * 4, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=2, ibn='a',
        #                                reduction=self.reduction)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self.construct_fc_layer(self.out_channels[3], self.out_channels[3], dropout_p=None)
        self.cn_modules = []
        self.initialize()
        if self.cnsn_type is not None and 'cn' in self.cnsn_type:
            # self.active_num = active_num
            assert self.active_num > 0
            print('active_num: {}'.format(self.active_num))
            self.cn_num = len(self.cn_modules)
            assert self.cn_num > 0
            print('cn_num: {}'.format(self.cn_num))

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            # elif isinstance(m, nn.BatchNorm2d):
            #    init.constant_(m.weight, 1)
            #    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, CrossNorm):
                self.cn_modules.append(m)
                
    def _enable_cross_norm(self):
        active_cn_idxs = np.random.choice(self.cn_num, self.active_num, replace=False).tolist()
        assert len(set(active_cn_idxs)) == self.active_num
        # print('active_cn_idxs: {}'.format(active_cn_idxs))
        for idx in active_cn_idxs:
            self.cn_modules[idx].active = True
            
    # def make_layer(self, blocks, in_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction):
    #     strides = [stride] + [1] * (blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction))
    #         self.inplane = planes
    #     return nn.Sequential(*layers)
    
    def make_layer(self, blocks, in_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction):
        layers = []
        layers += [blocks[0](in_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction)]

        return nn.Sequential(*layers)
    
    def construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # x_last = self.avgpool(x3)

        # x_last = x_last.view(x_last.size(0), -1)
        
        x0 = self.conv(x)
        x0 = self.pool(x0)
        x1 = self.osse1(x0)
        x1 = self.tran1(x1)
        x2 = self.osse2(x1)
        x2 = self.tran2(x2)
        x3 = self.osse3(x2)
        x3 = self.tran3(x3)

        x3 = self.global_avgpool(x3)
        x3 = x3.view(x3.size(0), -1)
        x_last = self.fc(x3)

        return x1, x2, x3, x_last
        
# =================================== feature extract module osECAIBN_cnsn ===================================
class Feature_Extract_Net_OSECAIBN_cnsn(nn.Module):
    def __init__(self):
        super(Feature_Extract_Net_OSECAIBN_cnsn, self).__init__()
        self.n_size = 3
        self.reduction = 4
        # self.channel = 3  # 如果选择序列order有改变，要修改哦

        self.in_channels = 3 # 如果选择序列order有改变，要修改哦
        self.out_channels = [16, 64, 96, 128]
        # self.inplane = 16
        # self.block = SEIBN_cnsn_OSBlock
        self.blocks = [[ECAIBN_cnsn_OSBlock, ECAIBN_cnsn_OSBlock], [ECAIBN_cnsn_OSBlock, ECAIBN_cnsn_OSBlock], [ECAIBN_cnsn_OSBlock, ECAIBN_cnsn_OSBlock]]
        
        self.active_num = 2
        self.pos = 'post' # pre/post/identity
        self.beta = None
        self.crop = 'neither'
        self.cnsn_type = 'cn'
        print('**********************************************')
        print('ResNet34 with oseca, ibn, cnsn...')

        self.conv = nn.Conv2d(self.in_channels, self.out_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_channels[0])
        self.relu = nn.ReLU(inplace=True)
        
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.osse1 = self.make_layer(self.blocks[0], self.out_channels[0], self.out_channels[1], self.pos, self.beta, self.crop, self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        self.tran1 = nn.Sequential(Conv1x1(self.out_channels[1], self.out_channels[1]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse2 = self.make_layer(self.blocks[1], self.out_channels[1], self.out_channels[2], self.pos, self.beta, self.crop, self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        self.tran2 = nn.Sequential(Conv1x1(self.out_channels[2], self.out_channels[2]),
                                    nn.AvgPool2d(2, stride=2))
        self.osse3 = self.make_layer(self.blocks[2], self.out_channels[2], self.out_channels[3], self.pos, self.beta, self.crop, self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        self.tran3 = Conv1x1(self.out_channels[3], self.out_channels[3])
        
        if self.beta is not None:
            print('beta: {}'.format(self.beta))

        if self.crop is not None:
            print('crop mode: {}'.format(self.crop))
        # self.layer1 = self._make_layer(self.block, self.inplane, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=1, ibn='a', reduction=self.reduction)
        # self.layer2 = self._make_layer(self.block, self.inplane * 2, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=2, ibn='a',
        #                               reduction=self.reduction)
        # self.layer3 = self._make_layer(self.block, self.inplane * 4, blocks=self.n_size, pos=self.pos, beta=self.beta, crop=self.crop, cnsn_type=self.cnsn_type, stride=2, ibn='a',
        #                                reduction=self.reduction)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self.construct_fc_layer(self.out_channels[3], self.out_channels[3], dropout_p=None)
        self.cn_modules = []
        self.initialize()
        if self.cnsn_type is not None and 'cn' in self.cnsn_type:
            # self.active_num = active_num
            assert self.active_num > 0
            print('active_num: {}'.format(self.active_num))
            self.cn_num = len(self.cn_modules)
            assert self.cn_num > 0
            print('cn_num: {}'.format(self.cn_num))

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            # elif isinstance(m, nn.BatchNorm2d):
            #    init.constant_(m.weight, 1)
            #    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, CrossNorm):
                self.cn_modules.append(m)
                
    def _enable_cross_norm(self):
        active_cn_idxs = np.random.choice(self.cn_num, self.active_num, replace=False).tolist()
        assert len(set(active_cn_idxs)) == self.active_num
        # print('active_cn_idxs: {}'.format(active_cn_idxs))
        for idx in active_cn_idxs:
            self.cn_modules[idx].active = True
            
    # def make_layer(self, blocks, in_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction):
    #     strides = [stride] + [1] * (blocks - 1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction))
    #         self.inplane = planes
    #     return nn.Sequential(*layers)
    
    def make_layer(self, blocks, in_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction):
        layers = []
        layers += [blocks[0](in_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction)]
        for i in range(1, len(blocks)):
            layers += [blocks[i](out_channels, out_channels, pos, beta, crop, cnsn_type, stride, ibn, reduction)]

        return nn.Sequential(*layers)
    
    def construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # x_last = self.avgpool(x3)

        # x_last = x_last.view(x_last.size(0), -1)
        
        x0 = self.conv(x)
        x0 = self.pool(x0)
        x1 = self.osse1(x0)
        x1 = self.tran1(x1)
        x2 = self.osse2(x1)
        x2 = self.tran2(x2)
        x3 = self.osse3(x2)
        x3 = self.tran3(x3)

        x3 = self.global_avgpool(x3)
        x3 = x3.view(x3.size(0), -1)
        x_last = self.fc(x3)

        return x1, x2, x3, x_last


