#-*- coding:utf-8 -*-
from __future__ import division 
from __future__ import absolute_import 
from __future__ import with_statement

import os
import time
import numpy as np
import argparse
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torchvision import transforms
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

from model import resnet
from model.trainer import fit,fit1
from model.metrics import AccumulatedAccuracyMetric
from utils.utils import extract_embeddings, plot_embeddings
from senet.se_resnet import se_resnet20#model = se_resnet20(num_classes=10, reduction=16)

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


parser.add_argument('-c', '--config', default='configs/who_config.json')
classes=('0','1')


args = parser.parse_args()

# Create model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)


## python liver.py -c configs/transfer_config.json
from data_loader.mri_t2wi import MRIORCT,MRIANDCT
from data_loader.datasets import SiameseMRI, TripletMRI
from utils.config import get_args, process_config
from utils.utils import printData
config = process_config(args.config)

"""
triplet_train_dataset = TripletMRI(train_dataset) # Returns pairs of images and target same/different
triplet_test_dataset = TripletMRI(test_dataset)
"""
# printData(test_dataset, type='normal')

# Set up data loaders

# batch_size = 32
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from model.cifar_networks import TripletNet
from model.losses import TripletLoss
from model.losses import OnlineTripletLoss
# Strategies for selecting triplets within a minibatch
from utils.utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector 
from utils.utils import extract_embeddings, plot_embeddings

avgtrain=[]
avgval=[]
final=[]
avgacc=[]
avgspec=[]
avgsens=[]
avgauu=[]
times=1

Sens=[]
Prec=[]
F1=[]
cnf_mat=[]
#cnf_mat2=np.array([[0, 0,0,0], [0, 0,0,0], [0, 0,0,0], [0, 0,0,0]])
cnf_mat2 = np.zeros((2, 2))
#cnf_mat2=np.array([[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]])
#cnf_mat2=np.array([[0, 0], [0, 0]])

avgmax=[]

for i in range(times):
    flag=False
    batch_size = 40
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    # Load data
    print('Create the data generator.')
    train_dataset = MRIORCT(config, train = True,classify = False,MRI = False)#MRI = False -- CT   
    test_dataset = MRIORCT(config, train = False,classify = False,MRI = False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    #res_model = resnet.resnet20(8,num_features = 64, num_classes = config.classes)
    res_model = se_resnet20(num_classes=64, in_channel = 3,reduction=16)


    
    margin=0.2#0.4
    res_model.cuda()
    loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin)).cuda()
    
    lr = 1e-3#-3
    n_epochs = 100 # 100 --triplet --90
    log_interval = 40    #log_interval = 100
    optimizer = torch.optim.Adam(res_model.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.98, last_epoch=-1)

    # lr = 1e-2 
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,35,50,75,100, 150])
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
    plot_embeddings(train_embeddings_cl, train_labels_cl, classes=["NotMVI","MVI"], save_tag = 'CTtrain11')
    val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
    plot_embeddings(val_embeddings_cl, val_labels_cl, classes=["NotMVI","MVI"], save_tag = 'CTtest11')


    c,d,f,Sens1, Prec1, F11, cnf_mat1,g=fit(train_loader, test_loader, res_model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,flag=flag)

    flag=True
    train_dataset = MRIORCT(config, train = True,classify = True,MRI = False)   
    test_dataset = MRIORCT(config, train = False,classify = True,MRI = False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    loss_fn = nn.NLLLoss().cuda() 
    lr1 = 1e-2#-2
    
    n_epochs = 50
    """
    for param in res_model.parameters():
        param.requires_grad = False
    """
    model1=resnet.ClassificationNet1(res_model,64, config.classes)
    #model1=res_model
    model1.cuda()
    #print(model1.embedding_net.conv1.weight)
    """
    for k,v in model1.named_parameters():
        print(k)
    """
    optimizer1 = torch.optim.SGD(model1.parameters(), lr1, momentum=0.9, weight_decay=5e-4)
    #optimizer1 = torch.optim.SGD([model1.linear1.weight,model1.linear2.weight,model1.linear3.weight,model1.linear4.weight], lr1, momentum=0.9, weight_decay=5e-4) 
    """
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1,'max',factor=0.35,verbose=True,patience=3)#min_lr=0.00001
    a,b,e,Sens1, Prec1, F11, cnf_mat1,k=fit1(train_loader, test_loader, model1, loss_fn, optimizer1, scheduler1, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()],flag=flag)
    """
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[25,40,50])#[25,40,50]--70 [20,35,50,100]
    a,b,e,Sens1, Prec1, F11, cnf_mat1,k=fit(train_loader, test_loader, model1, loss_fn, optimizer1, scheduler1, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()],flag=flag)

    #print(model1.embedding_net.conv1.weight)
    Sens.append(max(Sens1))
    Prec.append(max(Prec1))
    F1.append(max(F11))
    cnf_mat.append(cnf_mat1)
    for i in cnf_mat1:
        #print(i.shape)
        cnf_mat2 += i
    #print Sens, Prec, F1, cnf_mat
    avgtrain.append(a)
    avgval.append(b)
    final.append(e)
    avgmax.append(k)

    from utils.eval import validate
    validate(test_loader, model1.cuda(), nn.CrossEntropyLoss().cuda())
    validate(train_loader, model1.cuda(), nn.CrossEntropyLoss().cuda())
    torch.save(model1.state_dict(),'./CTbranch.pth')
avgtrain1=0
avgval1=0
for s in avgtrain:
    avgtrain1+=int(s)
for d in avgval:
    avgval1+=int(d)
u=max(avgtrain)
v=max(avgval)

# # Set up data loaders
# batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
plot_embeddings(train_embeddings_cl, train_labels_cl, classes=["NotMVI","MVI"], save_tag = 'CTtrain')
val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
plot_embeddings(val_embeddings_cl, val_labels_cl, classes=["NotMVI","MVI"], save_tag = 'CTtest')

print(cnf_mat2)
print("avgtrainacc for "+str(times)+" times "+str(avgtrain1/times))
print("avgtrainacc for "+str(times)+" times "+str(avgtrain1/times))
print("avgvalacc for "+str(times)+" times "+str(avgval1/times))
print("maxtrainacc for "+str(times)+" times "+str(u))
print("maxvalacc for "+str(times)+" times "+str(v))
print("--------------------------------")
print("avgmax for "+str(times)+" times "+str(sum(avgmax)/times))
avgmax1=np.array(avgmax)
print("avgmax std for "+str(times)+" times "+str(np.std(avgmax1)))
print("maxfinalvalacc for "+str(times)+" times "+str(max(final)))
print("avgfinalvalacc for "+str(times)+" times "+str(sum(final)/times))
for i in range(len(final)):
    final[i]/=100
nfinal=np.array(final)
print("var for "+str(times)+" times "+str(np.std(nfinal)))

print("Sens "+str(times)+" times "+str(sum(Sens)/len(Sens)))
print("Prec "+str(times)+" times "+str(sum(Prec)/len(Prec)))
print("F1 "+str(times)+" times "+str(sum(F1)/len(F1)))

Sens=np.array(Sens)
Prec=np.array(Prec)
F1=np.array(F1)
print("Sens std "+str(times)+" times "+str(np.std(Sens)))
print("Prec stf"+str(times)+" times "+str(np.std(Prec)))
print("F1 std"+str(times)+" times "+str(np.std(F1)))
