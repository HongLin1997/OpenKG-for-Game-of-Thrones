# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:42:12 2019

@author: admin
"""

"""Main script for ADDA."""

import params
from core import eval_src, train_src, eval_tgt
from utils import  init_model, init_random_seed, collate_fn
from importlib import import_module
import datasets, torch, os,pickle
#from sampler import ImbalancedDatasetSampler

from torch.utils.data import DataLoader

torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    
    x = import_module('models.' + params.model_name)
    config = x.Config(params.num_classes, params.pad_size,
                      params.dropout_linear, params.dropout_other)
        
    if torch.cuda.is_available():
        torch.cuda.set_device(params.num_gpu)
    # init random seed (ensure the same results we get at each time)
    init_random_seed(params.manual_seed)

    # load dataset
    DataModel = getattr(datasets, params.dataset + 'Data')
    data = DataModel(os.path.join(params.data_root, params.source_train))
    src_data_loader = DataLoader(data, 
                                 batch_size=params.batch_size, 
                                 shuffle=True, 
                                 num_workers=params.num_workers, 
                                 collate_fn=collate_fn)
    
    data = DataModel(os.path.join(params.data_root, params.source_validate))
    src_data_loader_validate = DataLoader(data, 
                                          batch_size=params.batch_size, 
                                          shuffle=True, 
                                          num_workers=params.num_workers, 
                                          collate_fn=collate_fn)
    
    data = DataModel(os.path.join(params.data_root, params.source_test))
    src_data_loader_test = DataLoader(data, 
                                      batch_size=params.batch_size, 
                                      shuffle=True, 
                                      num_workers=params.num_workers, 
                                      collate_fn=collate_fn)
    
    # load models
    src_encoder = x.Model(config).to(config.device)
    tgt_encoder = x.Model(config).to(config.device)
    classifier = x.Classifier(config).to(config.device)
    config = x.Config(params.tgt_num_classes, params.pad_size,
                      params.dropout_linear, params.dropout_other)
    tgt_classifier = x.Classifier(config).to(config.device)
    
    # initial model
    src_encoder = init_model(net=src_encoder, 
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=classifier, 
                                restore=params.src_classifier_restore)
    tgt_encoder = init_model(net=tgt_encoder, 
                             restore=params.tgt_encoder_restore)
    tgt_classifier = init_model(net=tgt_classifier, 
                                restore=params.tgt_classifier_restore)
    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        if os.path.exists(params.checkpoints_pretrain):
            checkpoints = pickle.load(open(params.checkpoints_pretrain,'rb'))
        else:
            print("no checkpoint in %s!"%params.checkpoints_pretrain)
            checkpoints = None
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader,
            src_data_loader_validate, checkpoints)
            
    # eval source model
    #print("=== Evaluating source classifier for source domain ===")
    #eval_src(src_encoder, src_classifier, src_data_loader_test)
    
########################## evaluate on unseen target ##########################
    
    data = DataModel(os.path.join(params.target_dataroot, params.target_train))
    tgt_data_loader_train = DataLoader(data, 
                                       batch_size=params.batch_size, 
                                       shuffle=True, 
                                       num_workers=params.num_workers, 
                                       collate_fn=collate_fn)
    
    data = DataModel(os.path.join(params.target_dataroot, params.target_validate))
    tgt_data_loader_dev = DataLoader(data, 
                                       batch_size=params.batch_size, 
                                       shuffle=True, 
                                       num_workers=params.num_workers, 
                                       collate_fn=collate_fn)
    
    data = DataModel(os.path.join(params.target_dataroot, params.target_test))
    tgt_data_loader_eval = DataLoader(data, 
                                      batch_size=params.batch_size, 
                                      shuffle=False, 
                                      num_workers=params.num_workers, 
                                      collate_fn=collate_fn)
    
    if not tgt_encoder.restored:
        print("no tgt_encoder_restored!")
        print("initialize target encoder with the parameters of the source!")
        init_with_src = dict([(k, 
                               v.clone()) for (k,v) in src_encoder.state_dict().items()])
                
        tgt_encoder.load_state_dict(init_with_src)
            
        
    if not (tgt_encoder.restored and tgt_classifier.restored and
            params.tgt_model_trained):
        if os.path.exists(params.checkpoints_adapt):
            checkpoints = pickle.load(open(params.checkpoints_adapt,'rb'))
        else:
            print("no checkpoint in %s!"%params.checkpoints_adapt)
            checkpoints = None
        tgt_encoder, tgt_classifier = train_src(
            tgt_encoder, tgt_classifier, tgt_data_loader_train,
            tgt_data_loader_dev, checkpoints, tgt_flag=True)
            
    print("=== Evaluating source classifier for target domain bc ===")
    eval_tgt(tgt_encoder, tgt_classifier, tgt_data_loader_eval)
    
    