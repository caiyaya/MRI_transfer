#-*- coding:utf-8 -*-
from __future__ import division 
from __future__ import absolute_import 
from __future__ import with_statement

import os
import time
import argparse
import torch
import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
from torchvision.datasets import MNIST
from torchvision import transforms
import torchvision
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

# Device configuration
cuda = torch.cuda.is_available()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                     

parser = argparse.ArgumentParser("""Image classifical!""")
parser.add_argument('--path', type=str, default='./data/cifar10/',
                    help="""image dir path default: './data/cifar10/'.""")
parser.add_argument('--epochs', type=int, default=50,
                    help="""Epoch default:50.""")
parser.add_argument('--batch_size', type=int, default=256,
                    help="""Batch_size default:256.""")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="""learing_rate. Default=0.0001""")
parser.add_argument('--num_classes', type=int, default=10,
                    help="""num classes""")
parser.add_argument('--model_path', type=str, default='./model/',
                    help="""Save model path""")
parser.add_argument('--model_name', type=str, default='cifar10.pth',
                    help="""Model name.""")
parser.add_argument('--display_epoch', type=int, default=5)

args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    # transforms.RandomHorizontalFlip(p=0.50),  # 有0.75的几率随机旋转
    # transforms.ColorJitter(brightness=1, contrast=2, saturation=3, hue=0),  # 给图像增加一些随机的光照
    transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
])

classes=('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
# Load data
train_dataset = torchvision.datasets.CIFAR10(root=args.path,
                                              transform=transform,
                                              download=True,
                                              train=True)


test_dataset = torchvision.datasets.CIFAR10(root=args.path,
                                             transform=transform,
                                             download=True,
                                             train=False)
                                           
# ###########################################################
# # ############################# Method 01：Softmax
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up the network and training parameters
# from model.cifar_networks import EmbeddingNet, ClassificationNet
# from model.metrics import AccumulatedAccuracyMetric
# from model.trainer import fit

# embedding_net = EmbeddingNet(in_channel=3, out_num=2)
# model = ClassificationNet(embedding_net, input_num=2, n_classes=10)
# if cuda:
    # model.cuda()
# loss_fn = torch.nn.NLLLoss()
# lr = 1e-2
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 20
# log_interval = 100                           
                            
# fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])                          
                                                                         
# # Save the model checkpoint
# torch.save(model, args.model_path + args.model_name)
# print("Model save to {}.".format(args.model_path + args.model_name))

# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test')

# ###########################################################                            
# #################################### Method 02：SiameseNet 
# from data.cifar_datasets import SiameseMNIST                           
# siamese_train_dataset = SiameseMNIST(train_dataset) # Returns pairs of images and target same/different
# siamese_test_dataset = SiameseMNIST(test_dataset)
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up the network and training parameters
# from model.cifar_networks import EmbeddingNet, SiameseNet
# from model.metrics import AccumulatedAccuracyMetric
# from model.losses import ContrastiveLoss
# from model.trainer import fit

# from torch.optim import lr_scheduler
# import torch.optim as optim

# margin = 1.
# embedding_net = EmbeddingNet(in_channel=3, out_num=32)
# model = SiameseNet(embedding_net)
# if cuda:
    # model.cuda()
# loss_fn = ContrastiveLoss(margin)
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 30
# log_interval = 100

# fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test')



# ############################################################
# ######################################## Method 03：Triplet

# # Set up data loaders
# from data.cifar_datasets import TripletMNIST

# triplet_train_dataset = TripletMNIST(train_dataset) # Returns triplets of images
# triplet_test_dataset = TripletMNIST(test_dataset)
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up the network and training parameters
# from model.cifar_networks import EmbeddingNet, TripletNet
# from model.losses import TripletLoss
# from model.trainer import fit

# margin = 1.0
# embedding_net = EmbeddingNet(in_channel=3, out_num=2)
# model = TripletNet(embedding_net)
# if cuda:
#     model.cuda()
# loss_fn = TripletLoss(margin)
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 200
# log_interval = 100

# fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)


# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test')


# ###############################################################
# ################################### Method 04:pair selection
# from data.cifar_datasets import BalancedBatchSampler
# # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
# train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=25)
# test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=25)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
# online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# # Set up the network and training parameters
# from model.trainer import fit
# from model.cifar_networks import EmbeddingNet
# from model.losses import OnlineContrastiveLoss
# from utils.utils import AllPositivePairSelector, HardNegativePairSelector # Strategies for selecting pairs within a minibatch

# margin = 1.
# embedding_net = EmbeddingNet(in_channel=3, out_num=2)
# model = embedding_net
# if cuda:
    # model.cuda()
# loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 200
# log_interval = 100

# fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test')


#############################################################
############################## Method 05:triplet selection

from data.cifar_datasets import BalancedBatchSampler

# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=25)
test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=25)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
from model.trainer import fit
from model.cifar_networks import EmbeddingNet
from model.losses import OnlineTripletLoss
# Strategies for selecting triplets within a minibatch
from utils.utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector 
from model.metrics import AverageNonzeroTripletsMetric, AccumulatedAccuracyMetric

margin = 1.
embedding_net = EmbeddingNet(in_channel=3, out_num=2)
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, 
metrics=[AverageNonzeroTripletsMetric(), AccumulatedAccuracyMetric()])

# 绘图
from utils.utils import extract_embeddings, plot_embeddings
# Set up data loaders
batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train')
val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test')

# #########################################################################
# ######################################Method06: Pretrained ResNet
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # 预训练Resnet
# from model import resnet
# pretrain_dict = torch.load('model/resnet20_dict.pkl')
# res_model = resnet.resnet20(3,num_features = 64, num_classes = 10)
# res_model.load_state_dict(pretrain_dict)
# print(res_model)

# # 验证预训练模型
# from utils.eval import validate
# validate(test_loader, res_model.cuda(), nn.CrossEntropyLoss().cuda())
# validate(train_loader, res_model.cuda(), nn.CrossEntropyLoss().cuda())

# # #提取fc层中固定的参数
# # features_num = res_model.module.linear.in_features
# # print(features_num)
# # #修改类别为2
# # res_model.module.linear = nn.Linear(features_num, 2)

# # 绘图, 原始Resnet提取的特征和softmax分类器不同
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test')


# ##############################################################################
# #######################################Method07: Train ResNet ################
# # Set up data loaders
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up the network and training parameters
# from model.metrics import AccumulatedAccuracyMetric
# from model.trainer import fit

# from model import resnet
# num_features = 2
# res_model = resnet.resnet20(3, num_features = num_features, num_classes = 10)
# print(res_model)

# res_model.cuda()
# loss_fn = nn.NLLLoss().cuda()
# lr = 1e-2
# optimizer = torch.optim.SGD(res_model.parameters(), lr, momentum=0.9, weight_decay=5e-4)   
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150])
# n_epochs = 200
# log_interval = 100                           
# fit(train_loader, test_loader, res_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()])

# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings

# #linearWeights = res_model.state_dict()['linear.weight'].cpu().numpy()
# # linearBias = res_model.state_dict()['linear.bias'].cpu().numpy()
# #linearBias = None

# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
# save_tag = 'train'
# plot_embeddings(train_embeddings_cl, train_labels_cl,classes=classes, save_tag = save_tag)
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
# save_tag = 'test'
# plot_embeddings(val_embeddings_cl, val_labels_cl,classes=classes, save_tag = save_tag)

# # Save model
# torch.save(res_model.state_dict(), './experiment/cifar_resnet_'+str(num_features)+'_dict.pkl')


##########################################################################################                          
#################################### Method 08：Siamese Resnet  ##########################
# from data.cifar_datasets import SiameseMNIST                           
# siamese_train_dataset = SiameseMNIST(train_dataset) # Returns pairs of images and target same/different
# siamese_test_dataset = SiameseMNIST(test_dataset)
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up the network and training parameters
# from model.cifar_networks import SiameseNet
# from model.losses import ContrastiveLoss
# from model.trainer import fit
# from model import resnet
# res_model = resnet.resnet20(3,num_features = 2, num_classes = 10)
# print(res_model)
# model = SiameseNet(res_model).cuda()
# loss_fn = ContrastiveLoss(1.0).cuda()
# lr = 1e-2
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# # optimizer = torch.optim.SGD(model.parameters(), lr,
#                             # momentum=0.9,
#                             # weight_decay=5e-4)   
# # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                  # milestones=[100, 150])
# n_epochs = 30
# log_interval = 100                           
# fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[])

# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test')



# ################################################################################
# ######################################## Method 09：Triplet Resnet  ############
# # Set up data loaders
# from data.cifar_datasets import TripletMNIST
# triplet_train_dataset = TripletMNIST(train_dataset) # Returns triplets of images
# triplet_test_dataset = TripletMNIST(test_dataset)
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# # Set up the network and training parameters
# from model.cifar_networks import TripletNet
# from model.losses import TripletLoss
# from model.trainer import fit

# from model import resnet
# res_model = resnet.resnet20(3,num_features = 2, num_classes = 10)
# print(res_model)
# model = TripletNet(res_model).cuda()
# loss_fn = TripletLoss(1.1).cuda()
# lr = 1e-2
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
# scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)
# n_epochs = 200
# log_interval = 100
# fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train Triplet Resnet')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test Triplet Resnet')
# torch.save(res_model.state_dict(), 'Triplet Resnet_dict.pkl')


# #######################################################################
# ################################### Method 10: Resnet pair selection
# from data.cifar_datasets import BalancedBatchSampler
# # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
# train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=25)
# test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=25)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
# online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# # Set up the network and training parameters
# from model.trainer import fit
# from model.cifar_networks import EmbeddingNet
# from model.losses import OnlineContrastiveLoss
# from utils.utils import AllPositivePairSelector, HardNegativePairSelector # Strategies for selecting pairs within a minibatch

# from model import resnet
# res_model = resnet.resnet20(3,num_features = 2, num_classes = 10)

# #res_model = resnet.ResNet(resnet.BasicBlock, [3, 3, 3], 2,10)

# model = res_model.cuda()
# loss_fn = OnlineContrastiveLoss(1.0, HardNegativePairSelector())
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)
# n_epochs = 200
# log_interval = 100

# fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)

# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test')


# ######################################################################
# ############################## Method 11: Resnet triplet selection
# from data.cifar_datasets import BalancedBatchSampler
# # We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
# train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=10, n_samples=25)
# test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=10, n_samples=25)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
# online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# # Set up the network and training parameters
# from model.trainer import fit
# from model.cifar_networks import EmbeddingNet
# from model.losses import OnlineTripletLoss
# # Strategies for selecting triplets within a minibatch
# from utils.utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector 
# from model.metrics import AverageNonzeroTripletsMetric

# from model import resnet
# res_model = resnet.resnet20(3,num_features = 2, num_classes = 10)
# #res_model = resnet.ResNet(resnet.BasicBlock, [3, 3, 3], 2)
# model = res_model.cuda()
# loss_fn = OnlineTripletLoss(1.0, HardestNegativeTripletSelector(1.0))
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
# scheduler = lr_scheduler.StepLR(optimizer, 30, gamma=0.1, last_epoch=-1)
# n_epochs = 500
# log_interval = 50

# fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, 
# metrics=[AverageNonzeroTripletsMetric()])

# # 绘图
# from utils.utils import extract_embeddings, plot_embeddings
# # Set up data loaders
# batch_size = 256
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
# train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
# plot_embeddings(train_embeddings_cl, train_labels_cl, classes, save_tag = 'train')
# val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
# plot_embeddings(val_embeddings_cl, val_labels_cl, classes, save_tag = 'test')

# # Save model
# torch.save(model.state_dict(), './model/triplet_selection_dict.pkl')


# ##############################################################
# ######################## Origin PyTorch0.3.1版本代码样例
# from __future__ import print_function
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
# import torch._utils
# try:
    # torch._utils._rebuild_tensor_v2
# except AttributeError:
    # def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        # tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        # tensor.requires_grad = requires_grad
        # tensor._backward_hooks = backward_hooks
        # return tensor
    # torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
# cuda = torch.cuda.is_available()
# # Training settings
# parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    # help='input batch size for training (default: 64)')
# parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    # help='input batch size for testing (default: 1000)')
# parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    # help='number of epochs to train (default: 10)')
# parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    # help='learning rate (default: 0.01)')
# parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    # help='SGD momentum (default: 0.5)')
# parser.add_argument('--no-cuda', action='store_true', default=False,
                    # help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=1, metavar='S',
                    # help='random seed (default: 1)')
# parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    # help='how many batches to wait before logging training status')
# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()

# torch.manual_seed(args.seed) #为CPU设置种子用于生成随机数，以使得结果是确定的
# if args.cuda:
    # torch.cuda.manual_seed(args.seed)#为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。


# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# """加载数据。组合数据集和采样器，提供数据上的单或多进程迭代器
# 参数：
# dataset：Dataset类型，从其中加载数据
# batch_size：int，可选。每个batch加载多少样本
# shuffle：bool，可选。为True时表示每个epoch都对数据进行洗牌
# sampler：Sampler，可选。从数据集中采样样本的方法。
# num_workers：int，可选。加载数据时使用多少子进程。默认值为0，表示在主进程中加载数据。
# collate_fn：callable，可选。
# pin_memory：bool，可选
# drop_last：bool，可选。True表示如果最后剩下不完全的batch,丢弃。False表示不丢弃。
# """
# train_loader = torch.utils.data.DataLoader(
    # datasets.MNIST('./mnist', train=True, download=True,
                   # transform=transforms.Compose([
                       # transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   # ])),
    # batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
    # datasets.MNIST('./mnist', train=False, transform=transforms.Compose([
                       # transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   # ])),
    # batch_size=args.batch_size, shuffle=True, **kwargs)


# class Net(nn.Module):
    # def __init__(self):
        # super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)#输入和输出通道数分别为1和10
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)#输入和输出通道数分别为10和20
        # self.conv2_drop = nn.Dropout2d()#随机选择输入的信道，将其设为0
        # self.fc1 = nn.Linear(320, 50)#输入的向量大小和输出的大小分别为320和50
        # self.fc2 = nn.Linear(50, 10)

    # def forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))#conv->max_pool->relu
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))#conv->dropout->max_pool->relu
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x))#fc->relu
        # x = F.dropout(x, training=self.training)#dropout
        # x = self.fc2(x)
        # return F.log_softmax(x)

# model = Net()
# if args.cuda:
    # model.cuda()#将所有的模型参数移动到GPU上

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# def train(epoch):
    # model.train()#把module设成training模式，对Dropout和BatchNorm有影响
    # for batch_idx, (data, target) in enumerate(train_loader):
        # if args.cuda:
            # data, target = data.cuda(), target.cuda()
        # '''
        # Variable类对Tensor对象进行封装，会保存该张量对应的梯度，以及对生成该张量的函数grad_fn的一个引用。
        # 如果该张量是用户创建的，grad_fn是None，称这样的Variable为叶子Variable。
        # '''
        # data, target = Variable(data), Variable(target)
        # optimizer.zero_grad()
        # output = model(data)
        # loss = F.nll_loss(output, target)#负log似然损失
        # loss.backward()
        # optimizer.step()
        # if batch_idx % args.log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                # epoch, batch_idx * len(data), len(train_loader.dataset),
                # 100. * batch_idx / len(train_loader), loss.data[0]))

# def test(epoch):
    # model.eval()#把module设置为评估模式，只对Dropout和BatchNorm模块有影响
    # test_loss = 0
    # correct = 0
    # for data, target in test_loader:
        # if args.cuda:
            # data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        # output = model(data)
        # test_loss += F.nll_loss(output, target).data[0]#Variable.data
        # pred = output.data.max(1)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data).cpu().sum()

    # test_loss = test_loss
    # test_loss /= len(test_loader) # loss function already averages over batch size
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        # test_loss, correct, len(test_loader.dataset),
        # 100. * correct / len(test_loader.dataset)))


# for epoch in range(1, args.epochs + 1):
    # train(epoch)
    # test(epoch)
