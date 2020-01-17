"""Pre-train encoder and classifier for source dataset."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, params
from utils import make_variable, save_model, adjust_learning_rate
import time, pickle
from .EarlyStopping import EarlyStopping
from tqdm import tqdm

def train_src(encoder, classifier, data_loader, 
              src_data_loader_eval, checkpoints=None, tgt_flag=False):
    
    """Train classifier for source domain."""
    time_local = time.localtime(time.time())
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    print(dt)
    ####################
    # 1. setup network #
    ####################
     
    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()
    
    if tgt_flag:
        not_bert_lr=params.d_learning_rate
        num_classes = params.tgt_num_classes
    else:
        num_classes = params.num_classes
        not_bert_lr=params.c_learning_rate
    # setup criterion and optimizer
    if params.optimizer=='sgd':
        optimizer = optim.SGD(
                list(encoder.parameters()) + list(classifier.parameters()),
                lr=not_bert_lr, weight_decay=params.weight_decay)
            
    elif params.optimizer == 'adamw':
        from transformers import AdamW
        named_parameters = list(dict(encoder.named_parameters()).items())+\
                           list(dict(classifier.named_parameters()).items())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_params = [
                {
                        'params': [p for n, p in named_parameters if ('bert' not in n) ], 
                        'weight_decay': params.weight_decay,
                        'lr': not_bert_lr,
                        'ori_lr': not_bert_lr
                },
                {
                        'params': [p for n, p in named_parameters if ('bert' in n) and not any(nd in n for nd in no_decay)], 
                        'weight_decay': 0.01,
                        'lr': params.bert_learning_rate,
                        'ori_lr': params.bert_learning_rate
                },
                {
                        'params': [p for n, p in named_parameters if ('bert' in n) and any(nd in n for nd in no_decay)], 
                        'weight_decay': 0.0,
                        'lr': params.bert_learning_rate,
                        'ori_lr': params.bert_learning_rate
                }
        ]
        optimizer = AdamW(grouped_params, correct_bias=False)
    else:
        optimizer = optim.Adam(
                list(encoder.parameters()) + list(classifier.parameters()),
                lr=not_bert_lr, weight_decay=params.weight_decay)
        
    if params.warmup_step > 0:
        from transformers import get_linear_schedule_with_warmup
        training_steps = params.num_epochs_pre * len(data_loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=params.warmup_step, num_training_steps=training_steps)
    else:
        scheduler = None
        
    criterion = nn.CrossEntropyLoss()
    
    if checkpoints==None:
        checkpoints={}
        start_epoch = 0
        start_step = 0
    else:
        print("load checkpoints!")
        start_epoch = checkpoints['epoch']
        print('start_epoch： ', start_epoch)
        start_step = checkpoints['step']
        print('start_step ', start_step)
        optimizer.load_state_dict(checkpoints['optimizer'])
        encoder.load_state_dict(checkpoints['encoder'])
        classifier.load_state_dict(checkpoints['classifier'])
    
    ####################
    # 2. train network #
    ####################
    # start to evaluate from training_acc > 0.5 
    # and start to save from validation_acc > 0.45
    best_epoch = 0
    best_f1_ex = 0
    e_f1_ex = 0
    accuracy_ex = 0
    best_encoder = None
    best_classifier = None
    # early stop training when the loss in validation set is not decreased for k times
    early_stopping = EarlyStopping(params.early_stop_patient, 
                                   verbose=True)

    for epoch in range(params.num_epochs_pre):
        
        if epoch < start_epoch:
            print('epoch continue: ', epoch)
            continue
        
        print('epochs: ', epoch)
        correct = 0
        total = 0
        target_num = torch.zeros((1, num_classes))
        predict_num = torch.zeros((1, num_classes))
        acc_num = torch.zeros((1, num_classes))
        total_loss=0
        # a large learning rate is needed to be decayed
        if params.c_learning_rate>=1e-3 and scheduler == None:
            adjust_learning_rate(optimizer, params.c_learning_rate, 
                                 decay_rate=.5, epoch=epoch, critic_flag=False)
        
        for step, (data, labels) in enumerate(data_loader):
            if epoch < start_epoch and step<start_step:
                continue
            # make data and labels variable
            # data = make_cuda(data)
            labels = make_variable(labels) #.squeeze_()
            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(data))
            loss1 = criterion(preds, labels) 
            loss = loss1
            
            # optimize source classifier
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
                    
            total_loss += loss1
            
            _, predicted = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
    
            pre_mask = torch.zeros(preds.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)
            tar_mask = torch.zeros(preds.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)
            acc_mask = pre_mask*tar_mask
            acc_num += acc_mask.sum(0)

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                micro_r = acc_num[:,1:].sum(1)/target_num[:,1:].sum(1) 
                micro_p = acc_num[:,1:].sum(1)/predict_num[:,1:].sum(1) if predict_num[:,1:].sum(1)!=0 else 0
                micro_f1 = 2*micro_p*micro_r/(micro_r+micro_p) if acc_num[:,1:].sum(1)!=0 else 0
                micro_r= (micro_r.numpy()[0]).round(4) if type(micro_r)!=int else micro_r
                micro_p= (micro_p.numpy()[0]).round(4) if type(micro_p)!=int else micro_p
                micro_f1= (micro_f1.numpy()[0]).round(4) if type(micro_f1)!=int else micro_f1
                
                recall = acc_num/target_num
                precision = acc_num/predict_num
                F1 = 2*recall*precision/(recall+precision)
                accuracy = acc_num.sum(1)/target_num.sum(1)
                accuracy_ex = acc_num[:,1:].sum(1)/target_num[:,1:].sum(1)
                
                recall = (recall.numpy()[0]).round(4)
                precision = (precision.numpy()[0]).round(4)
                F1 = (F1.numpy()[0]).round(4)
                accuracy = (accuracy.numpy()[0]).round(4)
                accuracy_ex = (accuracy_ex.numpy()[0]).round(4)
                
                where_are_nan = np.isnan(precision)
                precision[where_are_nan] = 0
                where_are_nan = np.isnan(recall)
                recall[where_are_nan] = 0
                where_are_nan = np.isnan(F1)
                F1[where_are_nan] = 0
                
                print("Epoch [{}/{}] Step [{}/{}]: loss={:.5f}, loss_without_L2={:.5f}, accuracy = {:.5f}, \
                      precision = {:.5f}, recall = {:.5f}, F1 = {}, accuracy(excludeNone) = {:.5f}, \
                      precision(excludeNone) = {:.5f}, recall(excludeNone) = {:.5f}, F1(excludeNone) = {:.5f} \
                      micro_precision(excludeNone) = {:.5f}, micro_recall(excludeNone) = {:.5f}, micro_F1(excludeNone) = {:.5f} \n"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.item(),
                              loss1.item(),
                              accuracy,
                              precision.sum()/num_classes,
                              recall.sum()/num_classes,
                              F1.sum()/num_classes,
                              accuracy_ex,
                              precision[1:].sum()/(num_classes-1),
                              recall[1:].sum()/(num_classes-1),
                              F1[1:].sum()/(num_classes-1),
                              micro_p,
                              micro_r,
                              micro_f1))
        
        print("Epoch [{}/{}]: total_loss:{}"
              .format(epoch + 1,
                      params.num_epochs_shared,
                      total_loss/len(data_loader)))
        
        # eval model on test set
        if epoch>0 and \
           ((epoch + 1) % params.eval_step_pre == 0):
            #e_f1_ex, e_loss = eval_src(encoder, classifier, 
            #                                 data_loader, log_flag=False, 
            #                                 tgt_flag=tgt_flag)
            e_f1_ex, e_loss = eval_src(encoder, classifier, 
                                             src_data_loader_eval, 
                                             log_flag=False, tgt_flag=tgt_flag)
            early_stopping(-e_f1_ex) #e_loss
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("Early stopping")
                # 结束模型训练
                break
            
        # save model parameters
        if epoch>0 and \
           e_f1_ex > best_f1_ex and \
           ((epoch + 1) % params.save_step_pre == 0):
               best_f1_ex = e_f1_ex
               best_epoch = epoch+1
               best_encoder = dict([(k, 
                                     v.clone()) for (k,v) in encoder.state_dict().items()])
               best_classifier = dict([(k, 
                                     v.clone()) for (k,v) in classifier.state_dict().items()])
               
               save_model(encoder, "{}-source-encoder-{}.pt".format(params.model_name, epoch + 1))
               save_model(
                classifier, "{}-source-classifier-{}.pt".format(params.model_name, epoch + 1))
               checkpoints['epoch'] = epoch
               checkpoints['step'] = step
               checkpoints['optimizer'] = optimizer.state_dict()
               checkpoints['encoder'] = encoder.state_dict()
               checkpoints['classifier'] = classifier.state_dict()
               if tgt_flag:
                   pickle.dump(checkpoints, 
                               open(os.path.join(params.model_root, 
                                                'checkpoint_adapt.pkl'),'wb'))
               else:
                   pickle.dump(checkpoints, 
                               open(os.path.join(params.model_root, 
                                                'checkpoint_pretrain.pkl'),'wb'))
                   
    # # save final model
    print("update as the best one trained at epoch %s!"%best_epoch)
    encoder.load_state_dict(best_encoder)
    classifier.load_state_dict(best_classifier)
    #print("saving the best...")
    #pickle.dump(best_encoder, 
    #            open(os.path.join(params.model_root, 
    #                              'best_encoder.pkl'),'wb'))
    #pickle.dump(best_classifier, 
    #            open(os.path.join(params.model_root, 
    #                              'best_classifier.pkl'),'wb'))
    print("saving...")
    save_model(encoder, "{}-source-encoder-final.pt".format(params.model_name))
    save_model(classifier, "{}-source-classifier-final.pt".format(params.model_name))
    #checkpoints['epoch'] = epoch
    #checkpoints['step'] = step
    #checkpoints['optimizer'] = optimizer.state_dict()
    #checkpoints['encoder'] = encoder.state_dict()
    #checkpoints['classifier'] = classifier.state_dict()
    #pickle.dump(checkpoints, 
    #          open(os.path.join(params.model_root, 
    #                            'checkpoint_pretrain.pkl'),'wb'))
    time_local = time.localtime(time.time())
    dt = time.strftime("%Y-%m-%d %H:%M:%S",time_local)
    print(dt)
    return encoder, classifier


def eval_src(encoder, classifier, data_loader,log_flag=True, tgt_flag=False):
    if tgt_flag:
        num_classes = params.tgt_num_classes
    else:
        num_classes = params.num_classes
    """Evaluate classifier for source domain.""" 
    if log_flag:
        paths = params.model_root.split("/")
        for i in range(len(paths)):
            if i==0:
                continue
            if os.path.exists("/".join(paths[0:i]))==False:
                os.mkdir("/".join(paths[0:i]))
        if os.path.exists(params.model_root)==False:
            os.mkdir(params.model_root)
        log = open(os.path.join(params.model_root, 'in_domain_test_results.txt'),'w')
        log.write(str(encoder) +'\n')
        log.write(str(classifier) +'\n')
        
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    #acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    correct = 0
    total = 0
    target_num = torch.zeros((1,num_classes))
    predict_num = torch.zeros((1,num_classes))
    acc_num = torch.zeros((1,num_classes))
    predict_matric = []
    truth_matric = []
    # evaluate network
    features = []
    for (data, labels) in tqdm(data_loader):
        labels = make_variable(np.array(labels)) #.squeeze_()
            
        feature = encoder(data)
        features.append(np.array(feature.data.cpu()))
        
        preds = classifier(feature)
        loss += criterion(preds, labels).item()

        #pred_cls = preds.data.max(1)[1]
        #acc += pred_cls.eq(labels.data).cpu().sum()
        _, predicted = torch.max(preds.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
    
        pre_mask = torch.zeros(preds.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        if len(predict_matric) == 0:
            predict_matric = pre_mask
        else:
            predict_matric = torch.cat((predict_matric, pre_mask), 0)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(preds.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
        if  len(truth_matric) == 0:
            truth_matric = tar_mask
        else:
            truth_matric = torch.cat((truth_matric, tar_mask), 0)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)


    loss /= len(data_loader)
    
    micro_r = acc_num[:,1:].sum(1)/target_num[:,1:].sum(1)
    micro_p = acc_num[:,1:].sum(1)/predict_num[:,1:].sum(1) if predict_num[:,1:].sum(1)!=0 else 0
    micro_f1 = 2*micro_p*micro_r/(micro_r+micro_p) if acc_num[:,1:].sum(1)!=0 else 0
    micro_r= (micro_r.numpy()[0]).round(4) if type(micro_r)!=int else micro_r
    micro_p= (micro_p.numpy()[0]).round(4) if type(micro_p)!=int else micro_p
    micro_f1= (micro_f1.numpy()[0]).round(4) if type(micro_f1)!=int else micro_f1
    
    #acc /= len(data_loader.dataset)
    recall = acc_num/target_num
    precision = acc_num/predict_num
    F1 = 2*recall*precision/(recall+precision)
    accuracy = acc_num.sum(1)/target_num.sum(1)
    accuracy_ex = acc_num[:,1:].sum(1)/target_num[:,1:].sum(1)
                       
    recall = (recall.numpy()[0]).round(4)
    precision = (precision.numpy()[0]).round(4)
    F1 = (F1.numpy()[0]).round(4)
    accuracy = (accuracy.numpy()[0]).round(4)
    accuracy_ex = (accuracy_ex.numpy()[0]).round(4)
              
    where_are_nan = np.isnan(precision)
    precision[where_are_nan] = 0
    where_are_nan = np.isnan(recall)
    recall[where_are_nan] = 0
    where_are_nan = np.isnan(F1)
    F1[where_are_nan] = 0
    print("Evaluate: loss={:.5f}, accuracy = {:.5f}, \
           precision = {:.5f}, recall = {:.5f}, F1 = {:.5f}, accuracy(excludeNone) = {:.5f}, \
           precision(excludeNone) = {:.5f}, recall(excludeNone) = {:.5f}, F1(excludeNone) = {:.5f} \
           micro_precision(excludeNone) = {:.5f}, micro_recall(excludeNone) = {:.5f}, micro_F1(excludeNone) = {:.5f} \n"
           .format(loss,
                   accuracy,
                   precision.sum()/num_classes,
                   recall.sum()/num_classes,
                   F1.sum()/num_classes,
                   accuracy_ex,
                   precision[1:].sum()/(num_classes-1),
                   recall[1:].sum()/(num_classes-1),
                   F1[1:].sum()/(num_classes-1),
                   micro_p,
                   micro_r,
                   micro_f1))
    if log_flag:
        log.write("Test: loss={:.5f}, accuracy = {:.5f}, \
                   precision = {:.5f}, recall = {:.5f}, F1 = {:.5f}, accuracy(excludeNone) = {:.5f}, \
                   precision(excludeNone) = {:.5f}, recall(excludeNone) = {:.5f}, F1(excludeNone) = {:.5f} \
                   micro_precision(excludeNone) = {:.5f}, micro_recall(excludeNone) = {:.5f}, micro_F1(excludeNone) = {:.5f} \n"
                   .format(loss,
                           accuracy,
                           precision.sum()/num_classes,
                           recall.sum()/num_classes,
                           F1.sum()/num_classes,
                           accuracy_ex,
                           precision[1:].sum()/(num_classes-1),
                           recall[1:].sum()/(num_classes-1),
                           F1[1:].sum()/(num_classes-1),
                           micro_p,
                           micro_r,
                           micro_f1))
        
        log.write(str(precision)+'\n')
        log.write(str(recall)+'\n')
        log.write(str(F1)+'\n')
        #log.write(str(truth_matric)+'\n')
        #log.write(str(predict_matric)+'\n')
        log.close()
        
        features = np.concatenate(tuple(features))
        np.save(os.path.join(params.model_root, "indomain_test_feature.npy"),features)
        np.save(os.path.join(params.model_root, "indomain_predict_matric.npy"),predict_matric.data.numpy())
        np.save(os.path.join(params.model_root, "indomain_truth_matric.npy"),truth_matric.data.numpy())
    
    encoder.train()  
    classifier.train()
     
    return  micro_f1, loss
    #print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
