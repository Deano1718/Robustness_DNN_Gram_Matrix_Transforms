from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import json
from torch.utils.data import RandomSampler
from collections import OrderedDict

from torchvision import datasets, transforms
#from torch.utils.tensorboard import SummaryWriter

from models.wideresnet import *
from models.resnet import *

from create_data import *
from smoothers import *
from utils import *
from kernel_utils import *

from art.defences.preprocessor import TotalVarMin

from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2CarliniWagnerAttack, LinfBasicIterativeAttack, L2DeepFoolAttack, BoundaryAttack, FGSM, L2ProjectedGradientDescentAttack




parser = argparse.ArgumentParser(description='PyTorch CIFAR Evaluate Defenses')


parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for evaluation (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', nargs='+',default=[0.031], type=float,
                    help='perturbation')
parser.add_argument('--num-steps', nargs='+',default=[20], type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.1, type=float,
                    help='perturb step size in percentage of epsilon')
parser.add_argument('--c_cw', nargs='+',default=[0.01], type=float,
                    help='CW attack initial c')
parser.add_argument('--kappa_cw', nargs='+',default=[0.0], type=float,
                    help='CW attack kappa confidence')
parser.add_argument('--binary_cw', nargs='+',default=[20], type=int,
                    help='CW attack binary search steps')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='./model-cifar10-',
                    help='directory of model for saving attack results')
parser.add_argument('--model-epoch', default=100, type=int,
                    help='epoch of model pnt to load')
parser.add_argument('--wide',default=1, type=int,
                    help='uses a Wide ResNet')
parser.add_argument('--attacks', nargs='+', default=[])
parser.add_argument('--num-points',default=1000, type=int,
                    help='number of randomly drawn test points to evaluate')
parser.add_argument('--preprocess', default='',
                    help='valid choices Median_Filter or TVM')
parser.add_argument('--defended', default=1, type=int,
                    help='whether to apply a defense or preprocess')
parser.add_argument('--transform-defense', default=1, type=int,
                    help='whether to apply a transformation defense or not')
parser.add_argument('--BPDA', default=1, type=int,
                    help='conduct end-to-end attack using BPDA')
parser.add_argument('--clean', default=0, type=int,
                    help='evaluate processes or defenses on non-adversarial clean examples')
parser.add_argument('--norm-type', default='batch',
                    help='batch, layer, or instance')
parser.add_argument('--norm-learn', default=1, type=int,
                    help='whether normalization is learnable')
parser.add_argument('--kernel-layers', nargs='+', default=[1,2,3,4,5], type=int,
                    help='kernel layers to use for transform')
parser.add_argument('--l1-c1', default=30.0, type=float,
                    help='l1 limit on transform')
parser.add_argument('--linf-c2', default=0.02, type=float,
                    help='linf limit on transform')
parser.add_argument('--iterations', default=10, type=int,
                    help='transform iterations')
parser.add_argument('--poly', default=(0,1), type=tuple,
                    help='poly kernel settings')
parser.add_argument('--quorum_c3', default=9, type=int,
                    help='transform iterations')
parser.add_argument('--rho', default=0.1, type=float,
                    help='rho in RMSprop')
parser.add_argument('--cp-name', default='',
                    help='name of checkpoint')



args = parser.parse_args()

mf = Median_Filter

FUNCTION_MAP = {'Median_Filter' : mf}

FUNCTION_ARGS = {'Median_Filter' : {'kernel': 2, 'bpda': 0},
                'TVM': {'p': 0.6, 'bpda': 1} }

TRANSFORM_PARAMS = {'step' : 0.07,
                    'opt' : 'RMSprop',
                    'l1_c1' : args.l1_c1,
                    'linf_c2' : args.linf_c2,
                    'iterations' : args.iterations,
                    'poly_settings' : args.poly,
                    'kernel_layers' : args.kernel_layers,
                    'quorum_c3' : args.quorum_c3,
                    'rho': args.rho,
                    'kernel_weights' : [1e6]*10,
                    'rel_size' : 1.0,
                    'replay' : 1}

print (TRANSFORM_PARAMS)

kwargsUser = {}
kwargsUser['norm_type'] = args.norm_type
kwargsUser['norm_learn'] = args.norm_learn


# settings
if (args.wide):
    network_string = 'wideResNet'
else:
    network_string = 'ResNet18'

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

# setup data loader
transform_tensor = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_tensor)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_tensor)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, **kwargs)

#draw subsample of test points to evaluate against for all attacks/defenses
t_point = 0
X_eval = []
Y_eval = []

for batch_idx, data in enumerate(test_loader):

    t_point += args.batch_size

    if (t_point > args.num_points):
        break

    #clean data
    X, Y = data[0], data[1]
    X_eval.append(X.clone())
    Y_eval.append(Y.clone())

X_eval = torch.cat(X_eval)
Y_eval = torch.cat(Y_eval)

eval_set = CustomDataSet(X_eval,Y_eval)
eval_loader = torch.utils.data.DataLoader(eval_set,batch_size=args.batch_size,shuffle=True)


def eval_points(model, device, test_loader):
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

def get_class_samplers(x,y):

    #build class samplers from correctly classified training points
    class_loaders = []
    class_iters = []
    for c in range(len(x)):
        #print ("length of x_lists is ", len(x_lists[c]))
        class_set_ = CustomDataSet(x[c], y[c])
        #class_rand_samp = RandomSampler(class_set_, replacement=True, num_samples=1)
        class_rand_samp = RandomSampler(class_set_, replacement=True)
        class_loader_ = torch.utils.data.DataLoader(class_set_, sampler=class_rand_samp, batch_size=1)
        class_loaders.append(class_loader_)
        datanext = iter(class_loaders[c])
        class_iters.append(datanext)

    return class_iters

def get_samples(samplers, num_samp, num_class=10, process='',**kwargs):

    samples = []

    for i in range(num_samp):

        example_next = samplers[i % num_class].next()
        example, example_label = example_next[0], example_next[1]
        #print (torch.mean(example))

        if process:
            example = process(example,**kwargs)

        samples.append(example.clone())

    return samples





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
    model.eval()

    try:
        preprocess = FUNCTION_MAP[args.preprocess]
        preprocess_dict = FUNCTION_ARGS[args.preprocess]

    except:
        print ("preprocess argument not found, proceeding with no preprocessing")
        preprocess = ''
        preprocess_dict = {}

    if (args.defended):
        try:
            x_lists, y_lists = get_split_correct_points(train_loader, model, device, preprocess, **FUNCTION_ARGS[args.preprocess])
        except:
            x_lists, y_lists = get_split_correct_points(train_loader, model, device)

        
        class_samplers = get_class_samplers(x_lists,y_lists)

    result_path = os.path.join(model_dir, 'Attack_Eval.txt')


    if not os.path.exists(result_path):
        with open(result_path, 'a') as f:
            f.write("Attack\tClean\tDefended\tProcess\tTransformDef\tBPDA\teps\tsteps\tc_cw\tkappa_cw\tbinary_cw\tacc\tL2Distortion\tModel_Epoch\n")
        f.close()



    #note how arguments are passed
    #each string in args.attacks defines an attack and will consume one entry in each of the other settings lists
    for a, att in enumerate(args.attacks):

        _EPSILON = float(args.epsilon[a])

        if (att == 'CW'):
            attack = L2CarliniWagnerAttack(binary_search_steps=float(args.binary_cw[a]), steps=int(args.num_steps[a]), stepsize=float(args.step_size)*_EPSILON, 
                confidence=float(args.kappa_cw[a]), initial_const=float(args.c_cw[a]), abort_early=True)
        elif (att == 'BIM'):
            attack = LinfBasicIterativeAttack(rel_stepsize=float(args.step_size),steps=int(args.num_steps[a]))
        elif (att == 'DF'):
            attack = L2DeepFoolAttack()
        elif (att == 'LinfPGD'):
            print ("LinfPGD attack selected")
            attack = LinfPGD(rel_stepsize=float(args.step_size), steps=int(args.num_steps[a]))
        else:
            print ("ERROR")

        


        #track number of successful adversaries from attacker's viewpoint
        adv_sum = 0.0
        adv_count = 0.0
        acc = 0.0
        L2_dist = []

        for batch_idx, (data, target) in enumerate(eval_loader):

            print ("entering batch")
            data, target = data.to(device), target.to(device)

 
            #define the model to attack based on threat model
            if args.defended:

                transform_samples = []
                adv_samples = []

                #for cifar10 get 20 samples, 10 for attacker and 10 for defender
                if args.transform_defense:
                    transform_samples = get_samples(samplers=class_samplers, num_class=10, num_samp=20,process=preprocess,**preprocess_dict)
                    adv_samples = transform_samples[:10]
                    def_samples = transform_samples[10:]

                #print (TRANSFORM_PARAMS)
                if args.BPDA:
                    #give hyperparameter settings to attacker
                    attack_model = Classifier_BPDA(batched=1,transformed=args.transform_defense,neural_net=model,device=device,transform_img=adv_samples,
                                                      transform_layers_weights=TRANSFORM_PARAMS['kernel_weights'],
                                                      layer_inds=TRANSFORM_PARAMS['kernel_layers'], c=TRANSFORM_PARAMS['poly_settings'][0],d=TRANSFORM_PARAMS['poly_settings'][1],
                                                      limInf=TRANSFORM_PARAMS['linf_c2'],
                                                      lim1=TRANSFORM_PARAMS['l1_c1'],
                                                      baseStep=TRANSFORM_PARAMS['step'],transformIter=TRANSFORM_PARAMS['iterations'],
                                                      process = preprocess,
                                                      **preprocess_dict)
                else:
                    attack_model = model


            else:
                attack_model = model


            ###############
            #ADVERSARY GENERATION IF APPLICABLE


            attack_model.eval()
            fmodel = PyTorchModel(attack_model, bounds=(0,1), preprocessing=None)


            if (not args.clean):
                print ("generating adversaries")
                raw, X_new_torch, is_adv = attack(fmodel, data, target, epsilons=_EPSILON)
                print ("finished generating adversaries")

                #cur_adv_sum += torch.sum(is_adv).item()
                #cur_non_adv_sum += (len(target) - torch.sum(is_adv).item())
            else:
                print ("evaluating on non-adversarial clean data")
                X_new_torch = data.clone()

            ################


            #DEFENSE SECTION IF APPLICABLE
            if args.defended:

                
                if preprocess:
                    X2 = preprocess(X_new_torch.clone(),**preprocess_dict)
                else:
                    X2 = X_new_torch.clone()

                if (not args.transform_defense):

                    with torch.no_grad():
                        Z_X2 = model(X2)
                        Yp_X2 = Z_X2.data.max(dim=1)[1] 
                        delta_correct = torch.sum(Yp_X2==target).item()
                        acc += delta_correct
                        if (delta_correct < args.batch_size):
                            L2_dist.append(torch.linalg.norm((X_new_torch[Yp_X2!=target] - data[Yp_X2!=target]).view(data[Yp_X2!=target].shape[0],-1), ord=2,dim=1).cpu())

                else: 

                    with torch.no_grad():
                        #get initial prediction using bare network
                        Z_X2 = model(X2)
                        Yp_X2_initial = Z_X2.data.max(dim=1)[1] 

                        hard_vote = []
                        Yp_new = []

                        # N x 10 matrix for hard votes
                        for r in range(len(target)):
                            hard_vote.append([])
                            Yp_new.append(Yp_X2_initial[r])           
                            for _ in range(10):
                                hard_vote[r].append(0)

                        model_tups = []
                        
                    for m in range(len(def_samples)):
                    
                        temp_model_filt, transform_loss_filt, choices = build_defense(model,
                                                                        transform_img=def_samples[m],
                                                                        transform_layers_ind=TRANSFORM_PARAMS['kernel_layers'],
                                                                        transform_layers_weights=TRANSFORM_PARAMS['kernel_weights'],
                                                                        koff=TRANSFORM_PARAMS['poly_settings'][0],
                                                                        kpwr=TRANSFORM_PARAMS['poly_settings'][1])


                    #model_tups.append( [temp_model_filt, transform_loss_filt] )


                    #for m, mod in enumerate(model_tups):

                        input_img_batch =  X2.clone().detach()

                        X_transformed, transform_scores = batch_stylize(temp_model_filt,
                                    input_img_batch, 
                                    kernel_losses = transform_loss_filt,
                                    kernel_weight=[1.0,1.0,1.0,1.0,1.0,1.0],
                                    rel_size=0.0,
                                    iterations=TRANSFORM_PARAMS['iterations'], 
                                    lim_inf=TRANSFORM_PARAMS['linf_c2'],
                                    lim_1=TRANSFORM_PARAMS['l1_c1'],
                                    optOption=TRANSFORM_PARAMS['opt'], 
                                    momentOption=0.8, 
                                    baseStep=TRANSFORM_PARAMS['step'], 
                                    rho=TRANSFORM_PARAMS['rho'],
                                    device = device)

                        Z_final = model(X_transformed)
                        Yp_final = Z_final.data.max(dim=1)[1]

                        for t in range(len(target)):
                            if (Yp_X2_initial[t].item() != Yp_final[t].item()):
                                hard_vote[t][Yp_final[t].item()] += 1

                    for c in range(len(target)):
                        if max(hard_vote[c]) >= TRANSFORM_PARAMS['quorum_c3']:
                            Yp_new[c] = hard_vote[c].index(max(hard_vote[c]))   # no control for ties technically, but rarely if ever occurs.  No advantage to either side anyways.
                        else:
                            Yp_new[c] = Yp_X2_initial[c]
                        
                        if Yp_new[c] == target[c]:
                            acc += 1.0
                        else:
                            if (not args.clean):
                                L2_dist.append(torch.linalg.norm((X_new_torch[c] - data[c]).view(data[c].shape[0],-1), ord=2, dim=1).cpu())

            else:
            #if not defended at all
                if preprocess:
                    X2 = preprocess(X_new_torch.clone(),**preprocess_dict)
                else:
                    X2 = X_new_torch.clone()

                with torch.no_grad():
                    Z_X2 = model(X2)
                    Yp_X2 = Z_X2.data.max(dim=1)[1] 
                    delta_correct = torch.sum(Yp_X2==target).item()
                    acc += delta_correct
                    if (not args.clean and delta_correct < args.batch_size):
                        L2_dist.append(torch.linalg.norm((X_new_torch[Yp_X2!=target] - data[Yp_X2!=target]).view(data[Yp_X2!=target].shape[0],-1), ord=2,dim=1).cpu())

        #after batches
        #accuracy for this attack
        print ("length of loader dataset is {} ".format(len(eval_loader.dataset)))
        acc /= len(eval_loader.dataset)
        print ("Accuracy: {0:6.3f}".format(acc))

        if (not args.clean):
            L2_dist = torch.mean(torch.cat(L2_dist)).item()
            print ("L2 Distortion: {0:6.3f}".format(L2_dist))
        else:
            L2_dist = 0.0

        #write to results file
        with open(result_path, 'a') as f:
            f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(
                att,
                args.clean,
                args.defended,
                args.preprocess,
                args.transform_defense,
                args.BPDA,
                args.epsilon[a],
                args.num_steps[a],
                args.step_size,
                args.c_cw[a],
                args.kappa_cw[a],
                args.binary_cw[a],
                args.model_epoch))
            f.write("\t{}\t".format(TRANSFORM_PARAMS['l1_c1']))
            f.write("{}\t".format(TRANSFORM_PARAMS['linf_c2']))
            f.write("{}\t".format(TRANSFORM_PARAMS['quorum_c3']))
            f.write("{}\t".format(TRANSFORM_PARAMS['iterations']))
            f.write("{}\t".format(TRANSFORM_PARAMS['kernel_layers']))
            f.write("{}\t".format(TRANSFORM_PARAMS['poly_settings']))
            f.write("{}\t".format(TRANSFORM_PARAMS['rho']))
            f.write("{}\t".format(args.seed))
            f.write("{0:6.3f}\t".format(acc))
            f.write("{0:6.3f}\n".format(L2_dist))

        f.close()



if __name__ == '__main__':
    main()
