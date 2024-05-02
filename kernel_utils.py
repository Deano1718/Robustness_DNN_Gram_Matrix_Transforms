import torch
import torch.nn as nn
import torch.nn.functional as F
import smoothers
import scipy
import scipy.special



class Classifier_BPDA(nn.Module):
    def __init__(self, batched, transformed, neural_net, device, transform_img = [],layer_inds=[], transform_layers_weights = [], c=0, d=1, limInf=-1.0, lim1=-1.0, 
                 baseStep=0.02, transformIter = 20, process='',BPDAprocess=False,**kwargs):
        super(Classifier_BPDA, self).__init__()

        self.batched = batched
        self.transformed = transformed
        self.neural_net = neural_net
        self.device = device
        self.transform_img = transform_img
        self.layer_inds = layer_inds
        self.transform_layers_weights = transform_layers_weights
        self.c = c
        self.d = d 
        self.limInf=limInf
        self.lim1=lim1
        self.baseStep=baseStep
        self.transformIter=transformIter
        self.process = process
        self.BPDAprocess = BPDAprocess
        self.layer_inds = layer_inds
        self.kwargs = kwargs
        #self.__dict__.update(kwargs)

        #

        
        #self.layer_names = ['conv1','block1','block2','block3','block4']        
        #self.layer_names = ['conv1','relu1','layer1','layer2','layer3','layer4']



        #self.tvm = TotalVarMin(prob=0.5, norm=2, lamb=0.08, max_iter=5, clip_values=(0.0,1.0),verbose=False)
        #if (self.transformed):
        #    self.kmodTups = self.populate_models()
        #self.kernel_tens, self.corr_tens = self.populate_models()

        # model_to_load = ('best_model0_ep148.pt','Final_ResNet18_TVMOnly_150ep',
        #          'ResNet18','CIFAR47K','std','AdversarialData_1026_01_27_07','ResNet18_Stats_1026','ResNet18TVMOnly')

        


    def populate_models(self):

        # (model, styles, contents)
        model_tups = []


        for m in range(len(self.transform_img)):

            temp_model_filt, transform_loss_filt, choices = build_defense(self.neural_net,
                                                                transform_img=self.transform_img[m],
                                                                transform_layers_ind=self.layer_inds,
                                                                transform_layers_weights=self.transform_layers_weights,
                                                                koff=self.c,
                                                                kpwr=self.d)
                                                                

            model_tups.append( [temp_model_filt, transform_loss_filt] )

        return model_tups

    def apply_transform(self,inp):

        filter_samples = []

        logits_out = 0.0

        for m in range(len(self.transform_img)):
            #print ("m is ", m)

            #content_img_batch = inp.clone()
            
            temp_model_filt, transform_loss_filt, choices = build_defense(self.neural_net,
                                                                transform_img=self.transform_img[m],
                                                                transform_layers_ind=self.layer_inds,
                                                                transform_layers_weights=self.transform_layers_weights,
                                                                koff=self.c,
                                                                kpwr=self.d)
                                                                
                                                                
            input_img_batch = inp.clone().detach()     #no detach because input_img_batch will be modified


            X_cleaned_filt, style_scores_filt = batch_stylize(temp_model_filt,
                                        input_img_batch, 
                                        kernel_losses = transform_loss_filt,
                                        kernel_weight=[1.0,1.0,1.0,1.0,1.0,1.0],
                                        rel_size=0.0,
                                        iterations=self.transformIter, 
                                        lim_inf=self.limInf,
                                        lim_1=self.lim1,
                                        optOption='RMSprop', 
                                        momentOption=0.8, 
                                        baseStep=self.baseStep, 
                                        rho=0.1,
                                        device = self.device)
            
            out = inp.clone()  #100,3,32,32
            out.data = X_cleaned_filt.data #100,3,32,32

            #print ("out shape ", out.shape)
            logit = self.neural_net(out)   #100,10
            #print ("logit shape ",logit.shape)
            logit_mag = torch.linalg.norm(logit,2, dim=1).unsqueeze(1).expand_as(logit)
            #print (logit_mag.shape)
            logit = logit / logit_mag
            logits_out += logit

                
        return logits_out


    def forward(self, x):
        """Perform forward."""

        if (self.process != ''):
            if (self.BPDAprocess):
                smooth_x = self.process(x, **self.kwargs)
                x1 = x.clone()
                x1.data = smooth_x.data             #for non-differentiable processes
            else:
                x1 = self.process(x, **self.kwargs)   #differentiable with PyTorch functions
        else:
            x1 = x



        if (self.transformed):
            with torch.enable_grad():
                x1 = self.apply_transform(x1)


        return x1


def kernel_matrix(inp,off,pwr):
    a, b, c, d = inp.size()  # for batch size 1

    features = inp.view(a * b, c * d).clone()  

    G = (torch.mm(features, features.t()) + off)**pwr  # compute the kernel matrix

    #normalize by number of terms
    return G.div(scipy.special.comb(c*d,pwr))

def kernel_matrix_batch(inp,off,pwr):
    a, b, c, d = inp.size()  # a=batch size

    #bmm implementation
    batch_features = inp.view(a, b, c*d)
    kmatrices = (torch.bmm(batch_features, torch.transpose(batch_features,1,2)) + off)**pwr

    return kmatrices.div(scipy.special.comb(c*d,pwr))

class KernelCalc(nn.Module):

    def __init__(self, input_feature_batch ,off=0, pwr=1):
        super(KernelCalc, self).__init__()
        self.off = off
        self.pwr = pwr

        #extract kernel matrices from self.target
        self.target = kernel_matrix_batch(input_feature_batch,self.off,self.pwr).detach()


    def forward(self, inp):

        return inp


class KernelLossBatchImg(nn.Module):

    def __init__(self, target_feature, loss_weight=1, off=0, pwr=1):
        super(KernelLossBatchImg, self).__init__()

        self.loss_weight = loss_weight
        self.off = off
        self.pwr = pwr

        self.target = kernel_matrix_batch(target_feature,self.off,self.pwr).detach()

    def forward(self, inp):

        G = kernel_matrix_batch(inp, off=self.off, pwr=self.pwr)   

        self.loss = self.loss_weight*F.mse_loss(G, 
                                            self.target.clone().expand_as(G), 
                                            reduction = 'none')

        self.loss = torch.mean(self.loss,(1,2))

        return inp


class KernelLossBatch(nn.Module):

    def __init__(self, kernel_comp, loss_weight=1, off=0, pwr=1, corr=0, corrTens = torch.empty(1,1)):
        super(KernelLossBatch, self).__init__()

        self.loss_weight = loss_weight
        self.loss_type = loss_type

        self.kernel_comp = kernel_comp.detach()      #may be without batch dimension if using expand
        self.corr = corr
        #corrTens shoud be preprocessed to not have entries with zero
        #corrTens should be preprocessed for the batch such that each batch class gets correct
        #correction matrix
        self.corrTens = corrTens.detach()
        self.off = off
        self.pwr = pwr

    def forward(self, inp):

        G = kernel_matrix_batch(inp, off=self.off, pwr=self.pwr)   #BATCH DIMENSION MUST BE GREATER THAN 1 !!!!

        # self.diags = torch.eye(G.shape[1],device=DEVICE).expand_as(G)
        # self.nondiags = (torch.ones((G.shape[1],G.shape[1]),device=DEVICE) - torch.eye(G.shape[1],device=DEVICE)).expand_as(G)

        if (not self.corr):

            if (self.loss_type == 'mse'):
                self.loss = self.loss_weight*F.mse_loss(G, 
                                                self.kernel_comp.clone().expand_as(G), 
                                                reduction = 'none')
            else:
                print ("ERROR")

        else:

            if (self.loss_type == 'mse'):
                self.loss = self.loss_weight*F.mse_loss((G/self.corrTens).clone(), 
                                                (self.kernel_comp.clone().expand_as(G)/self.corrTens).clone(), 
                                                reduction = 'none')
            else:
                print ("ERROR")

        self.loss = torch.mean(self.loss,(1,2))

        return inp



def build_defense(cnn, transform_img,
                                transform_layers_ind = [],
                                transform_layers_weights=[1.0,1.0,1.0,1.0,1.0,1.0],
                                    koff=0,
                                    kpwr=1,
                                    kernel_comp_list = [],
                                    correction_list = [],
                                    corrected = 0,
                                    def_type = "img"):
    
    cnn.eval()

    kernel_losses = []
    children = []
    choices = []

    model = nn.Sequential()

    cur_ind = 0
    weight_ind = 0

    for name, layer in cnn.named_children():
        children.append(name)


    for name, layer in cnn.named_children():

        #note responsibility of caller to correctly know layers and deposit desired indices
        #function will output the choices to show result
        if (cur_ind in transform_layers_ind):
            choices.append(children[cur_ind])

            if (def_type=="img"):
                target_feature = model(transform_img).detach()
                kernel_loss = KernelLossBatchImg(target_feature, loss_weight=transform_layers_weights[weight_ind],off=koff,pwr=kpwr)
            elif (def_type=="matrix"):
                kernel_loss = KernelLossBatch(kernel_comp_list[weight_ind], loss_weight=transform_layers_weights[weight_ind],off=koff,pwr=kpwr,
                    corr=corrected,corrTens=correction_list[weight_ind])
            else:
                print ("ERROR in building defense model")



            weight_ind += 1
                
            model.add_module("KL_{}".format(name), kernel_loss)
            kernel_losses.append(kernel_loss)

        cur_ind +=1


    for i in range(len(model) - 1, -1, -1):
        if (isinstance(model[i], KernelLossBatchImg) or isinstance(model[i], KernelLossBatch)):
            break

        model = model[:(i + 1)]
                
     
    return model, kernel_losses, choices



def batch_stylize(network, input_img, kernel_losses, kernel_weight=[], rel_size=0.01, iterations=20, lim_inf=125.0/255, lim_1=-1.0, optOption='RMSprop',
                  momentOption=0.8, baseStep=0.3, rho=0.1, device=''):
        
    network.eval()

    kernel_score_list = []
    for h in range(len(kernel_losses)):
        kernel_score_list.append([])

    a, b, c, d = input_img.size() 

    last_mat_tracker = torch.zeros_like(input_img)
    last_projmat_tracker = torch.ones_like(input_img)
    baseStep_matrix = baseStep*torch.ones_like(input_img)

    input_img_orig = input_img.clone().detach()

    global_noise_data = rel_size*lim_inf*torch.rand([a,b,c,d],dtype=torch.float, device=device)

    for i in range(iterations):

        noise_batch = global_noise_data[0:a].clone().detach().requires_grad_(True).to(device)

        _inputs = input_img_orig + noise_batch
        _inputs.clamp_(0.0, 1.0)

        #no output for "defended network"
        network(_inputs)
        
        kernel_score = 0.0

        for k, kl in enumerate(kernel_losses):
            kernel_score += kernel_weight[k]*kl.loss
            if (i % 5 == 0):
                kernel_score_list[k].append(kl.loss)
        
        loss = kernel_score
        
        loss.backward(gradient=torch.ones_like(loss))  
        
        with torch.no_grad():

            gradients_unscaled = noise_batch.grad
            #grad_mag = gradients_unscaled.view(gradients_unscaled.shape[0], -1).norm(2, dim=-1)
            #gradients = baseStep*gradients_unscaled  / grad_mag.view(-1, 1, 1, 1)

            if (optOption== 'RMSprop'):
                last_mat_tracker = last_mat_tracker*rho + (gradients_unscaled**2)*(1.0-rho)
                alpha = baseStep_matrix / (1e-8 + (last_mat_tracker**0.5))  #normally 1e-8
                perts = alpha*gradients_unscaled
                global_noise_data[0:a] -= perts.data
                #print (torch.mean(perts[0]))
                if (lim_1 > 0.0):
                    global_noise_data[0:a] = project_onto_l1_ball(global_noise_data[0:a], lim_1)

                global_noise_data.clamp_(-lim_inf, lim_inf)
                noise_batch.grad.zero_()
            
            elif (optOption == 'SGD'):
                perts = (baseStep*gradients_unscaled) + (momentOption * last_mat_tracker)
                last_mat_tracker = perts
                global_noise_data[0:a] -= perts.data
                print (torch.max(perts[0]))
                if (lim_1 > 0.0):
                    global_noise_data[0:a] = project_onto_l1_ball(global_noise_data[0:a], lim_1)
                    
                global_noise_data.clamp_(-lim_inf, lim_inf)
                noise_batch.grad.zero_()
            else:
                print ("ERROR")

    transformed_imgs = (input_img_orig + global_noise_data[0:a].clone().detach()).clone().detach()
    transformed_imgs.clamp_(0.0,1.0)

    
    return transformed_imgs, kernel_score_list


def project_onto_l1_ball(x, eps):
    """
    Compute Euclidean projection onto the L1 ball for a batch.
    
      min ||x - u||_2 s.t. ||u||_1 <= eps
    
    Inspired by the corresponding numpy version by Adrien Gaidon.
    
    Parameters
    ----------
    x: (batch_size, *) torch array
      batch of arbitrary-size tensors to project, possibly on GPU
      
    eps: float
      radius of l-1 ball to project onto
    
    Returns
    -------
    u: (batch_size, *) torch array
      batch of projected tensors, reshaped to match the original
    
    Notes
    -----
    The complexity of this algorithm is in O(dlogd) as it involves sorting x.
    
    References
    ----------
    [1] Efficient Projections onto the l1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
    """
    original_shape = x.shape
    x = x.view(x.shape[0], -1)
    mask = (torch.norm(x, p=1, dim=1) < eps).float().unsqueeze(1)
    mu, _ = torch.sort(torch.abs(x), dim=1, descending=True)
    cumsum = torch.cumsum(mu, dim=1)
    arange = torch.arange(1, x.shape[1] + 1, device=x.device)
    rho, _ = torch.max((mu * arange > (cumsum - eps)) * arange, dim=1)
    theta = (cumsum[torch.arange(x.shape[0]), rho.cpu() - 1] - eps) / rho
    proj = (torch.abs(x) - theta.unsqueeze(1)).clamp(min=0)
    x = mask * x + (1 - mask) * proj * torch.sign(x)
    return x.view(original_shape)
    
  




