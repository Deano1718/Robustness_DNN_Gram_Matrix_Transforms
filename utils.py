import torch
import torch.nn as nn
import torch.nn.functional as F
import smoothers



def get_split_correct_points(dataloader, model, device, process='', **kwargs):
    
    correct_lists = []
    correct_lab_lists = []
    
    for i in range(10):
        correct_lists.append([])
        correct_lab_lists.append([])
    
    with torch.no_grad(): # tell Pytorch not to build graph in this section
        for batch_idx, data in enumerate(dataloader):

            X, Y = data[0].to(device), data[1].to(device)

            if process:
                X = process(X,**kwargs)

            Z = model(X)

            #make prediction, do not need softmax
            Yp = Z.data.max(dim=1)[1]  # get the index of the max for each batch sample
                                       # Z.data.max(dim=1) returns two tensors, [0] is values, [1] is indices

            filtered_data = X[Yp==Y].clone().detach()
            filtered_labels = Y[Yp==Y].clone().detach()

            
            for j in range(10):
                correct_lists[j].append(filtered_data[filtered_labels==j].clone().detach())
                correct_lab_lists[j].append(filtered_labels[filtered_labels==j].clone().detach())

            
        for L in range(len(correct_lists)):
            correct_lists[L] = torch.cat(correct_lists[L])
            correct_lab_lists[L] = torch.cat(correct_lab_lists[L])
                

    return correct_lists, correct_lab_lists
