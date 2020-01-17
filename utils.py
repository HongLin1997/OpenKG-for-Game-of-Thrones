"""Utilities for ADDA."""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import params

def set_requires_grad(model, requires_grad=True):
    for name, param in model.named_parameters():
        # do not change the requires_grad of word embeddings
        if 'word_embs' in name:
            continue
        param.requires_grad = requires_grad

def loop_iterable(iterable):
    while True:
        yield from iterable

def collate_fn(batch):
    '''
    custom for DataLoader
    '''
    data, label = zip(*batch)
    return data, label

def one_hot_label(label):
    tensor = np.zeros((len(label), params.num_classes))
    for i, l in enumerate(label):
        tensor[i,l]=1
    return Variable(torch.from_numpy(tensor))

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        return Variable(torch.LongTensor(tensor).cuda(), volatile=volatile)
    else:
        return  Variable(torch.LongTensor(tensor), volatile=volatile)
        

def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def adjust_learning_rate(optimizer, init_lr, decay_rate=.5, 
                         epoch=0, critic_flag= False):
    lr = init_lr * (decay_rate ** (epoch // 2))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    return lr

def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore, method='xavier', exclude='embedding', seed=123):
    
    """Init models with cuda and weights."""
    # init weights of model
    # net.apply(init_weights)
    for name, w in net.named_parameters():
        if 'bert' not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.uniform_(w, 0, 0)
                #nn.init.constant_(w, 0)
            else:
                pass
            
    # restore model weights
    if restore is not None and os.path.exists(restore):
        if torch.cuda.is_available():
            net.load_state_dict(torch.load(restore))
        else:
            net.load_state_dict(torch.load(restore, map_location=torch.device('cpu')))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))
    else:
        print("No files in {}".format(os.path.abspath(restore)))
    
    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    paths = params.model_root.split("/")
    for i in range(len(paths)):
        if i==0:
            continue
        if os.path.exists("/".join(paths[0:i]))==False:
            os.mkdir("/".join(paths[0:i]))
    
    if os.path.exists(params.model_root)==False:
        os.mkdir(params.model_root)
            
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))
