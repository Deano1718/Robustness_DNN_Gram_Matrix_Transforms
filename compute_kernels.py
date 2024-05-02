from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import json
from torch.utils.data import RandomSampler, DistributedSampler
from collections import OrderedDict
import numpy as np

from torchvision import datasets, transforms
#from torch.utils.tensorboard import SummaryWriter

from models.wideresnet import *
from models.resnet import *

from create_data import *
from smoothers import *
from utils import *
from kernel_utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Evaluate Defenses')


parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                    help='input batch size for evaluation (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./model-cifar10-',
                    help='directory of model for saving attack results')
parser.add_argument('--model-epoch', default=100, type=int,
                    help='epoch of model pnt to load')
parser.add_argument('--wide',default=0, type=int,
                    help='uses a Wide ResNet')
parser.add_argument('--num-points',default=1000, type=int,
                    help='number of randomly drawn test points to evaluate')
parser.add_argument('--preprocess', default='',
                    help='valid choices Median_Filter or TVM')
parser.add_argument('--norm-type', default='batch',
                    help='batch, layer, or instance')
parser.add_argument('--kernel-layers', nargs='+', default=[1,2,3,4,5], type=int,
                    help='kernel layers to use for transform')
parser.add_argument('--poly', default=(0,1), type=tuple,
                    help='poly kernel settings')
parser.add_argument('--cp-name', default='',
                    help='name of checkpoint')
parser.add_argument('--dataset', default="CIFAR10",
                    help='name of checkpoint')



args = parser.parse_args()



kwargsUser = {}
kwargsUser['norm_type'] = args.norm_type
kwargsUser['norm_learn'] = args.norm_learn


# settings
if (args.wide):
    network_string = 'wideResNet'
else:
    network_string = 'ResNet18'

model_dir = args.model_dir
data_dir = "Kernel_Data"


data_path = os.path.join(model_dir, data_dir)

if (!os.path.exists(data_path)):
    os.mkdir(data_path)

# with open('{}/commandline_args.txt'.format(model_dir), 'a') as f:
#     json.dump(args.__dict__, f, indent=2)
# f.close()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)

# setup data loader
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
])

if (args.dataset == "CIFAR10"):
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_tensor)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, **kwargs)
elif (args.dataset == "CIFAR100"):
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_tensor)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, **kwargs)
elif (args.dataset == "IMAGENET"):
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
else:
    print ("ERROR GETTING DATA")

    #testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_tensor)
    #test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs)


def main():

    torch.cuda.empty_cache()
    # init model, ResNet18() can be also used here for training
    if args.wide==34:
        model = WideResNet(depth=34,**kwargsUser).to(device)
    elif args.wide==28:
        model= WideResNet(depth=28,**kwargsUser).to(device)
    elif args.wide==50:
        model = ResNet50(**kwargsUser).to(device)
    else:
        model = ResNet18(**kwargsUser).to(device)



    #load model and set to eval mode
    if args.cp_name == '':
        model_pnt = torch.load('{}/model-{}-epoch{}.pt'.format(model_dir,network_string,args.model_epoch))
        model.load_state_dict(model_pnt,strict=False)
    else:
        model_pnt = torch.load('{}/{}'.format(model_dir,args.cp_name))
        if ('state_dict' in model_pnt.keys()):
            model_pnt = model_pnt['state_dict']
        #print (model_pnt.keys())
        new_state_dict = OrderedDict()
        for k, v in model_pnt.items():
            if "module" in k:
                name = k[7:]
                new_state_dict[name] = v
            else:
                name = k
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict,strict=False)


    #model.load_state_dict(model_pnt)
    model.eval().to(device)

    x_lists, y_lists = get_split_correct_points(train_loader, model, device)

    cd_combo = [(0,1),(1,2),(1,3)]
    set_prefix = ['Cifar10Clean']


    for cd_tup in cd_combo:
        mean_grams = []
        var_grams = []

        for i in range(len(x_lists)):
            print ("class ", i)                  

            initialized = 0

            mean_grams.append([])  #list will hold list of mean grams for class i

            class_set_ = CustomDataSet(x_lists[i], y_lists[i])
            class_loader_ = DataLoader(class_set_, batch_size=args.batch_size, shuffle=True)


            for batch_idx, data in enumerate(class_loader_):     #8 indents

                X, Y = data[0].to(device), data[1].cpu()

                with torch.no_grad():


                    model, kernel_losses, choices = build_defense(model_eval,
                                                                 transform_img=X,
                                                                 transform_layers_ind = [0,1,2,3,4,5],
                                                                 transform_layers_weights=[1.0]*10,
                                                                 koff=cd_tup[0],
                                                                 kpwr=cd_tup[1])

                                        ###
                    if (not initialized):
                        #populates first batch of grams for each layer for class i
                        initialized=1
                        for s, sl in enumerate(kernel_losses):
                            mean_grams[i].append(sl.target.clone().sum(dim=0).cpu())   # goes to mean_grams[0][0], [0][1] for class i

                    else:
                        #mean_grams is a list (classes) of list (layers)
                        for s, sl in enumerate(kernel_losses):
                            mean_grams[i][s].add_(sl.target.clone().sum(dim=0).cpu())

            for m in mean_grams[i]:        #for each layer evaluated
                m.div_(len(class_loader_.dataset))
                #print (torch.max(m))
                #print (torch.min(m))
                #print ("gram shape", m.shape)

            print ("choices ", choices)


            initialized = 0
            var_grams.append([])

            ref_mean_grams = []
            for p in range(len([0,1,2,3,4,5])):
                ref_mean_grams.append(mean_grams[i][p].clone().unsqueeze(0).to(device))

            for batch_idx, data in enumerate(class_loader_):     #8 indents

                X, Y = data[0].to(device), data[1].cpu()

                with torch.no_grad():


                    model, kernel_losses, choices = build_defense(model_eval,
                                                                 transform_img=X,
                                                                 transform_layers_ind = [0,1,2,3,4,5],
                                                                 transform_layers_weights=[1.0]*10,
                                                                 koff=cd_tup[0],
                                                                 kpwr=cd_tup[1])

                                        ###
                    if (not initialized):
                        initialized=1
                        for s, sl in enumerate(kernel_losses):
                            var_grams[i].append(((sl.target - ref_mean_grams[s])**2).clone().sum(dim=0).cpu())

                    else:
                        for s, sl in enumerate(kernel_losses):
                            var_grams[i][s].add_(((sl.target - ref_mean_grams[s])**2).clone().sum(dim=0).cpu())

            for v in var_grams[i]:                  
                #this means I calculated sigma^2_i
                v.div_(len(class_loader_.dataset)).pow_(0.5)   #convert to std dev

                             
            for layer_lvl in range(len(mean_grams[i])):
                torch.save(mean_grams[i][layer_lvl],
                    os.path.join(data_path, 'MeanKernel_{}_cd{}{}_Class{}_Layer{}.pt'.format(set_prefix[0], cd_tup[0],cd_tup[1], i, layer_lvl)))
                torch.save(var_grams[i][layer_lvl],
                    os.path.join(data_path, 'DevKernel_{}_cd{}{}_Class{}_Layer{}.pt'.format(set_prefix[0], cd_tup[0],cd_tup[1], i, layer_lvl)))





