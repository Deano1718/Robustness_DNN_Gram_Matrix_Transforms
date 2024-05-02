import torch
import torch.nn as nn
import torch.nn.functional as F


def Median_Filter(tens, **kwargs):  #kernel=2, stride=1, quant=0):
    #returns a version of tens with median filter applied
    N, C, H, W = tens.size()

    #print (N,C,H,W)

    k = kwargs['kernel']

    if (H % 2 == 0):
        p2d = (0, 1, 0, 1) 
        out = F.pad(tens, p2d, "reflect", 0)


    unfold = nn.Unfold(kernel_size=(k,k),stride=1,dilation=1)
    output = unfold(out)


    output = output.view(N,C,k*k,-1).transpose(2,3)

    med = torch.median(output,dim=3)


    # if (kwargs[quant] == 0):
    #     med = torch.median(output,dim=3)
    # else:
    #     med = torch.quantile(output, q=0.5, dim=3)


    return med[0].reshape((N,C,H,W))
