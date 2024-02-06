import torch
import numpy as np
from model.transfer_model import *

def CDAN(input_list, domain_shape, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach() # 切断softmax的反传
    feature = input_list[0]
    if random_layer is None:
        # torch.bmm(a,b)：tensor a的size为(b,h,w)，tensor b的size为(b,w,h)，两个tensor的维度必须为3.
        # 因此需要unsqueeze增加维度计算
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    dc_target = torch.from_numpy(np.array([[1]] * domain_shape[0] + [[0]] * domain_shape[1])).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

# 引入平方对抗损失函数 和 类敏感的正则化项
def CDAN_New(input_list, domain_shape, ad_net, entropy=None, coeff=None, random_layer=None, lambda_c=0.1):
    softmax_output = input_list[1].detach() # 切断softmax的反传
    feature = input_list[0]
    if random_layer is None:
        # torch.bmm(a,b)：tensor a的size为(b,h,w)，tensor b的size为(b,w,h)，两个tensor的维度必须为3.
        # 因此需要unsqueeze增加维度计算
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    # 域分类目标
    dc_target = torch.from_numpy(np.array([[1]] * domain_shape[0] + [[0]] * domain_shape[1])).float().cuda()

    # 计算平方对抗损失
    bce_loss = nn.BCELoss(reduction='none')
    adv_loss = bce_loss(ad_out, dc_target)
    squared_adv_loss = torch.mean(adv_loss ** 2)

    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0 + torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0) // 2:] = 0
        source_weight = entropy * source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0) // 2] = 0
        target_weight = entropy * target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()

        # 原始return的
        weighted_loss = torch.sum(weight.view(-1, 1) * adv_loss) / torch.sum(weight).detach().item()

        # 基于样本的不确定性的加权正则
        class_sensitive_loss = lambda_c * (torch.sum(source_weight) + torch.sum(target_weight))
        total_loss = squared_adv_loss + class_sensitive_loss + weighted_loss

        return total_loss
    else:
        return nn.BCELoss()(ad_out, dc_target)

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy