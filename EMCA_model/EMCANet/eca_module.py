import torch
from torch import nn
from torch.nn.parameter import Parameter


class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        dilation: 膨胀率（kernel的间隔数量）
        modality: 模态数
    """
    def __init__(self, channel, k_size=3, modality=8):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化，b,c不变，输出为1*1
        # dilation = int((channel+1+(k_size - 1)//2-k_size-modality)/(k_size-1))+1+1
        # print(int((channel-modality)/(k_size-1))+1+1)
        # dilation的计算公式为论文公式5，
        # 前两个参数分别为输入和输出的通道数
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1)//2+1, dilation=int((channel-modality)/(k_size-1))+1+1, bias=False) #padding=(k_size - 1) // 2
        self.sigmoid = nn.Sigmoid()
        self.modality = modality

    def forward(self, x):
        # feature descriptor on the global spatial information
        b, c, h, w = x.size()
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # print(y.squeeze(-1).transpose(-1, -2).size())
        # torch.squeeze()压缩维度；torch.unsqueeze()扩展维度；torch.transpose交换一个tensor的两个维度
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print(x.size())
        # print(y.size())
        # Multi-scale information fusion
        y = self.sigmoid(y)  # 各个模态的权重向量
        # print(y.expand(b,self.modality,h,w).size())
        v = torch.rand((b, c, h, w)).cuda()
        for i in range(0, c):
            # print("i:",i)
            # print("c: ",int(i*self.modality//c))
            v[:, i, :, :] = y.expand(b, self.modality, h, w)[:, i*self.modality//c, :, :]
        """
        for i in range(0,c):
            #print("i:",i)
            #print("c: ",int(i*self.modality//c))
            x[:,i,:,:] = x[:,i,:,:]*y.expand(b,self.modality,h,w)[:,i*self.modality//c,:,:]
        """
        return x * v

