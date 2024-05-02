import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from smoothers import Median_Filter



class CustomDataSet(Dataset):   

    def __init__(self, Xdata, Ydata):
        self.Xdata = Xdata
        self.Ydata = Ydata
        
    def __len__(self):
        return len(self.Xdata)   
    
    def __getitem__(self, index):

        return self.Xdata[index], self.Ydata[index]



def compute_smooth_data(clean_dataset_loader, device, smoother='Median'):
        
    set_length = len(clean_dataset_loader.dataset)

    #first_batch = 1

    xdat = []
    ydat = []

    print ('Computing smoothed data')

    for batch_idx, data in enumerate(clean_dataset_loader):


        X, Y = data[0].to(device), data[1].to(device)


        if (smoother == 'Median'):
        	X_smooth = Median_Filter(X,kernel=2)
        else:
        	print ("ERROR in smoothing data")

        #ytemp = Y.unsqueeze(1).cpu()*torch.ones([len(Y), 1], dtype=torch.long)

        xdat.append(X_smooth.clone())
        ydat.append(Y.clone())

        # if (first_batch):
        #     x_dat = X_tvm_tens.clone().detach().cpu()
        #     #x_dat = torch.cat((x_dat,X_tvm_tens.clone()),0)
        #     y_dat = ytemp.clone().detach().cpu()
        #     #y_dat = torch.cat((y_dat,ytemp.clone()),0)
        #     first_batch = 0
        # else:
        #     x_dat = torch.cat((x_dat,X_tvm_tens.clone()),0)
        #     y_dat = torch.cat((y_dat,ytemp.clone()),0)


        #print ("{}/{}".format(batch_idx * X.size(0), set_length))
        if (batch_idx % 10 == 0):
       		print ('batch', batch_idx, 'complete')

    xdatcat = torch.cat(xdat)
    ydatcat = torch.cat(ydat)
        
    return xdatcat.cpu(), ydatcat.cpu()


def merge_data(dataset_loader1, dataset_loader2):
        
    #set_length = len(clean_dataset_loader.dataset)

    #first_batch = 1

    xdat = []
    ydat = []

    print ('Merging natural and smoothed data')


    for batch_idx, data in enumerate(dataset_loader1):

        X, Y = data[0].cpu(), data[1].cpu()

        xdat.append(X.clone())
        ydat.append(Y.clone())

        # if (first_batch):
        #     x_dat = X.clone().detach().cpu()
        #     y_dat = Y.unsqueeze(1).cpu()*torch.ones([len(Y), 1], dtype=torch.long)
        #     first_batch = 0
        # else:
        #     x_dat = torch.cat((x_dat,X.clone().detach().cpu()),0)
        #     y_dat = torch.cat((y_dat,Y.unsqueeze(1).cpu()*torch.ones([len(Y), 1], dtype=torch.long)),0)


        #print ("{}/{}".format(batch_idx * X.size(0), set_length))

    for batch_idx, data in enumerate(dataset_loader2):

        X, Y = data[0].cpu(), data[1].cpu()

        xdat.append(X.clone())
        ydat.append(Y.clone())

        if (batch_idx % 10 == 0):
       		print ('batch', batch_idx, 'complete')

        # x_dat = torch.cat((x_dat,X.clone().detach().cpu()),0)
        # y_dat = torch.cat((y_dat,Y.cpu()*torch.ones([len(Y), 1], dtype=torch.long)),0)


        #print ("{}/{}".format(batch_idx * X.size(0), set_length))

    xdatcat = torch.cat(xdat)
    ydatcat = torch.cat(ydat)
        
    return xdatcat, ydatcat