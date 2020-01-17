"""Test script to classify target data."""

import torch, os
import torch.nn as nn
import params
from utils import make_variable
import numpy as np
from tqdm import tqdm
import json

def eval_tgt(encoder, classifier, data_loader, source_flag = False):
    """Evaluation for target encoder by source classifier on target dataset."""
    log = open(os.path.join(params.model_root, 'test_results_%s.txt'%source_flag),'w')
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
    target_num = torch.zeros((1,params.num_classes))
    predict_num = torch.zeros((1,params.num_classes))
    acc_num = torch.zeros((1,params.num_classes))
    predict_matric = []
    truth_matric = []
    features = []
    
    x = []
    for (data, labels) in tqdm(data_loader):
        x += data
        labels = make_variable(labels) #.squeeze_()
        with torch.no_grad():
            feature = encoder(data)
        features.append(np.array(feature.data.cpu()))

        with torch.no_grad():
            preds = classifier(feature)
        loss += criterion(preds.detach(), labels).item()#.data[0]
        
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
        if len(truth_matric) == 0:
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
    #print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    print("Test: loss={:.5f}, accuracy = {:.5f}, \
           precision = {:.5f}, recall = {:.5f}, F1 = {:.5f}, accuracy(excludeNone) = {:.5f}, \
           precision(excludeNone) = {:.5f}, recall(excludeNone) = {:.5f}, F1(excludeNone) = {:.5f} \
           micro_precision(excludeNone) = {:.5f}, micro_recall(excludeNone) = {:.5f}, micro_F1(excludeNone) = {:.5f} \n"
           .format(loss,
                   accuracy,
                   precision.sum()/params.num_classes,
                   recall.sum()/params.num_classes,
                   F1.sum()/params.num_classes,
                   accuracy_ex,
                   precision[1:].sum()/(params.num_classes-1),
                   recall[1:].sum()/(params.num_classes-1),
                   F1[1:].sum()/(params.num_classes-1),
                   micro_p,
                   micro_r,
                   micro_f1))
    log.write("Test: loss={:.5f}, accuracy = {:.5f}, \
               precision = {:.5f}, recall = {:.5f}, F1 = {:.5f}, accuracy(excludeNone) = {:.5f}, \
               precision(excludeNone) = {:.5f}, recall(excludeNone) = {:.5f}, F1(excludeNone) = {:.5f}\
               micro_precision(excludeNone) = {:.5f}, micro_recall(excludeNone) = {:.5f}, micro_F1(excludeNone) = {:.5f} \n"
               .format(loss,
                       accuracy,
                       precision.sum()/params.num_classes,
                       recall.sum()/params.num_classes,
                       F1.sum()/params.num_classes,
                       accuracy_ex,
                       precision[1:].sum()/(params.num_classes-1),
                       recall[1:].sum()/(params.num_classes-1),
                       F1[1:].sum()/(params.num_classes-1),
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
    np.save(os.path.join(params.model_root, "source_%s_%s_test_feature.npy"%(str(source_flag), params.target_domain)),features)
    np.save(os.path.join(params.model_root, "source_%s_%s_predict_matric.npy"%(str(source_flag), params.target_domain)),predict_matric.data.numpy())
    np.save(os.path.join(params.model_root, "source_%s_%s_truth_matric.npy"%(str(source_flag), params.target_domain)),truth_matric.data.numpy())
    
    result=predict_matric.data.numpy()
    predict_not_none=[index for index, i in enumerate(result[:,0]) if i!=1]
    ccks_id2relation={}
    with open('../raw_data/ccks_IPRE/relation2id.txt','r',encoding='utf-8') as f:
        lines=f.readlines()
    for lin in lines:
        lin=lin.strip()
        ccks_id2relation[int(lin.split("\t")[1])]=lin.split("\t")[0]
    
    predict_notNone_result=[]
    for i in predict_not_none:
        print(i)
        x[i]['ccks_pred_label']=ccks_id2relation.get(list(result[i,:]).index(1))
        predict_notNone_result+=[x[i]]
    
    with open(os.path.join(params.model_root,
                           'source_%s_pred_notNoneCCKS_result.jsonl'%str(source_flag)),'w') as f:
        for item in predict_notNone_result:
            json.dump(item, f)
            f.write("\n")
            
    pred_triples=[]
    pred_triples_notNR=[]
    for item in predict_notNone_result:
        pred_triples+=[{'e1':item['h']['name'],
                        'r':item['ccks_pred_label'],
                        'e2':item['t']['name']}]
        if item['h']['tag'].startswith('nr') or \
        item['t']['tag'].startswith('nr'):
            continue
        pred_triples_notNR+=[{'e1':item['h']['name'],
                              'r':item['ccks_pred_label'],
                              'e2':item['t']['name']}]
    with open(os.path.join(params.model_root,
                           'source_%s_pred_notNoneCCKS_triples.jsonl'%str(source_flag)),'w') as f:
        for item in pred_triples:
            json.dump(item, f)
            f.write("\n")
    
    with open(os.path.join(params.model_root,
                           'source_%s_pred_notNoneCCKS_triples_notNR.jsonl'%str(source_flag)),'w') as f:
        for item in pred_triples_notNR:
            json.dump(item, f)
            f.write("\n")
      
    encoder.train()  
    classifier.train()
    return micro_f1 