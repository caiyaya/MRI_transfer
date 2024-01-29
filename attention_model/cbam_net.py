# ！！！！！！！！！！！！ 能跑通，但效果很差，就当个范例看就ok ！！！！！！！！！！！！！ #



# import torch
# import torch.nn as nn
# import torch.nn.init as init
#
#
#
#
# class ChannelAttention(nn.Module):
#     """
#     CBAM混合注意力机制的通道注意力
#     """
#
#     def __init__(self, in_channels, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(
#             # 全连接层
#             # nn.Linear(in_planes, in_planes // ratio, bias=False),
#             # nn.ReLU(),
#             # nn.Linear(in_planes // ratio, in_planes, bias=False)
#
#             # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
#             nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#         def forward(self, x):
#             avg_out = self.fc(self.avg_pool(x))
#             max_out = self.fc(self.max_pool(x))
#             out = avg_out + max_out
#             out = self.sigmoid(out)
#             return out * x
#
#
#
# class SpatialAttention(nn.Module):
#     """
#     CBAM混合注意力机制的空间注意力
#     """
#
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avg_out, max_out], dim=1)
#         out = self.sigmoid(self.conv1(out))
#         return out * x
#
#
#
#
# class CBAM(nn.Module):
#     """
#     CBAM混合注意力机制
#     """
#
#     def __init__(self, in_channels, ratio=16, kernel_size=3):
#         super(CBAM, self).__init__()
#         self.channelattention = ChannelAttention(in_channels, ratio=ratio)
#         self.spatialattention = SpatialAttention(kernel_size=kernel_size)
#
#     def forward(self, x):
#         x = self.channelattention(x)
#         x = self.spatialattention(x)
#         return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_att(x) * x
        out = self.spatial_att(out) * out
        return out



class CBAM_Net(nn.Module):
    def __init__(self, inplanes, num_class):
        super(CBAM_Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=64, kernel_size=3, padding=1)
        self.cbam1 = CBAM(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.cbam2 = CBAM(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.cbam3 = CBAM(256)

        self.fc = nn.Linear(256 * 16 * 16, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (32, 64, 128, 128)
        x = self.cbam1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (32, 64, 64, 64)
        x = F.relu(self.conv2(x))  # (32, 128, 64, 64)
        x = self.cbam2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (32, 128, 32, 32)
        x = F.relu(self.conv3(x))  # (32, 256, 32, 32)
        x = self.cbam3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (32, 256, 16, 16)

        x = x.view(x.size(0), -1)  # (32, 65536)
        x = self.fc(x)

        return x

