from __future__ import print_function
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import datetime
from datetime import datetime

from torchvision import datasets, transforms
#from torch.utils.tensorboard import SummaryWriter

from models.wideresnet import *
from models.resnet import *
from trades import trades_loss

from create_data import compute_smooth_data, merge_data, CustomDataSet

import smoothers


parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES + prdnn Adversarial Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-dir', default='./model-',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--smooth',default=True,
                    help='whether to compute and merge in pre-smoothed training data')
parser.add_argument('--wide',default=1, type=int,
                    help='uses a Wide ResNet')
parser.add_argument('--restart',default=0, type=int,
                    help='restart training, make sure to specify directory')
parser.add_argument('--restart-epoch',default=0,
                    help='epoch to restart from')
parser.add_argument('--norm-type', default='batch',
                    help='batch, layer, or instance')
parser.add_argument('--norm-learn', default=1, type=int,
                    help='whether normalization is learnable')
parser.add_argument('--adversarial', default=1, type=int,
                    help='adversarial training or not')
parser.add_argument('--dataset', default="CIFAR10",
                    help="which dataset")



args = parser.parse_args()

kwargsUser = {}
kwargsUser['norm_type'] = args.norm_type
kwargsUser['norm_learn'] = args.norm_learn

# settings

if (args.wide):
    network_string = 'wideResNet'
else:
    network_string = 'ResNet18'
    
def get_datetime():
    now = datetime.now()
    dt_string = now.strftime("%m%d_%H_%M_%S")
    return dt_string


if (not args.restart):
    model_dir = ("{}_{}_{}_epochs_{}_smooth_{}_beta_{}_{}".format(args.model_dir,args.dataset,network_string,args.epochs,args.smooth,args.beta,get_datetime()))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
else:
    model_dir = args.model_dir

with open('{}/commandline_args.txt'.format(model_dir), 'a') as f:
    json.dump(args.__dict__, f, indent=2)
f.close()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}
torch.cuda.empty_cache()

print ("cuda: ", use_cuda)
print ("smooth ", args.smooth)

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if (args.dataset == "CIFAR10"):
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
elif (args.dataset == "CIFAR100"):
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
else:
    print ("ERROR getting dataset")

if (args.smooth == True):
    xsmooth, ysmooth = compute_smooth_data(train_loader,device)
    smooth_set = CustomDataSet(xsmooth,ysmooth)
    smooth_loader = torch.utils.data.DataLoader(smooth_set,batch_size=200, shuffle=False)
    xm, ym = merge_data(train_loader,smooth_loader)
    merged_set = CustomDataSet(xm,ym) 
    train_loader = torch.utils.data.DataLoader(merged_set, batch_size=args.batch_size, shuffle=True, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    print ('Training model')
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        if (args.adversarial):
            loss = trades_loss(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer,
                           step_size=args.step_size,
                           epsilon=args.epsilon,
                           perturb_steps=args.num_steps,
                           beta=args.beta)
        else:
            logits = model(data)
            loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def eval_train(model, device, train_loader):
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    training_accuracy = correct / len(train_loader.dataset)
    return train_loss, training_accuracy


def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= (0.5*args.epochs):
        lr = args.lr * 0.1
    if epoch >= (0.75*args.epochs):
        lr = args.lr * 0.01
    # if epoch >= (0.9*args.epochs):
    #     lr = args.lr * 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # init model, ResNet18() can be also used here for training
    if args.wide:
        model = WideResNet(**kwargsUser).to(device)
    else:
        if (args.dataset=="CIFAR10"):
            model = ResNet18(10,**kwargsUser).to(device)
        elif (args.dataset=="CIFAR100"):
            model = ResNet18(100,**kwargsUser).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #writer = SummaryWriter()

    if (args.restart):
        model_pnt = torch.load('{}/model-{}-epoch{}.pt'.format(model_dir,network_string,args.restart_epoch))
        opt_pnt = torch.load('{}/opt-{}-checkpoint_epoch{}.tar'.format(model_dir,network_string,args.restart_epoch))
        model.load_state_dict(model_pnt)
        optimizer.load_state_dict(opt_pnt)

    for epoch in range(1+args.restart_epoch, args.epochs + 1 + args.restart_epoch):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        print('================================================================')
        loss_train, acc_train = eval_train(model, device, train_loader)
        loss_test, acc_test = eval_test(model, device, test_loader)
        print('================================================================')


        with open('{}/train_hist.txt'.format(model_dir), 'a') as f:
            f.write("{0:4.3f}\t{1:4.3f}\n".format(acc_train,acc_test))
        f.close()
        #writer.add_scalar("Loss/train", loss_train, epoch)
        #writer.add_scalar("Acc/train", acc_train, epoch)
        #writer.add_scalar("Loss/test", loss_test, epoch)
        #writer.add_scalar("Acc/test", acc_test, epoch)


        # save checkpoint
        # change file name here if it needs to be
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-{}-epoch{}.pt'.format(network_string,epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-{}-checkpoint_epoch{}.tar'.format(network_string,epoch)))
    #writer.flush()


if __name__ == '__main__':
    main()

