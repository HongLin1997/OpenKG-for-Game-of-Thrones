# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:33:14 2020

@author: admin
"""
import os, json

raw_data_path='raw_data/ccks_IPRE/'
output_data_path='datasets/ccks_IPRE/'

print("processing train/sent_relation_train.txt")
train_label = {}
train_multilabel=0
with open(os.path.join(raw_data_path,'train/sent_relation_train.txt'),'r',
          encoding='utf-8') as f:
    lines = f.readlines()
for lin in lines:
    lin=lin.strip().split("\t")
    try:
        train_label[lin[0]] = int(lin[1])
    except:
        train_multilabel+=1
        for l in lin[1].split(" "):
            train_label[lin[0]] = lin[1].split(" ")
    
print("processing train/sent_train_1.txt")
train = []
with open(os.path.join(raw_data_path,'train/sent_train_1.txt'),'r',
          encoding='utf-8') as f:
    lines = f.readlines()
for lin in lines:
    lin=lin.strip().split("\t")
    id_= lin[0]
    head = lin[1]
    tail = lin[2]
    tokens = lin[3].split(" ")
    head_index = tokens.index(head)
    tail_index = tokens.index(tail)
    train.append({'token':tokens,
                  'h':{'name':head, 'pos':[head_index, head_index+1]},
                  't':{'name':tail, 'pos':[tail_index, tail_index+1]},
                  'label':train_label.get(id_),
                  'id':id_})
    
print("processing train/sent_train_2.txt")
with open(os.path.join(raw_data_path,'train/sent_train_2.txt'),'r',
          encoding='utf-8') as f:
    lines = f.readlines()
for lin in lines:
    lin=lin.strip().split("\t")
    id_= lin[0]
    head = lin[1]
    tail = lin[2]
    tokens = lin[3].split(" ")
    head_index = tokens.index(head)
    tail_index = tokens.index(tail)
    if type(train_label.get(id_))==list:
        for l in train_label.get(id_):
            l=int(l)
            train.append({'token':tokens,
                         'h':{'name':head, 'pos':[head_index, head_index+1]},
                         't':{'name':tail, 'pos':[tail_index, tail_index+1]},
                         'label':l,
                         'id':id_})
    else:
        train.append({'token':tokens,
                  'h':{'name':head, 'pos':[head_index, head_index+1]},
                  't':{'name':tail, 'pos':[tail_index, tail_index+1]},
                  'label':train_label.get(id_),
                  'id':id_})

print("processing test/sent_relation_test.txt")    
test_label = {}
test_multilabel=0
with open(os.path.join(raw_data_path,'test/sent_relation_test.txt'),'r',
          encoding='utf-8') as f:
    lines = f.readlines()
for lin in lines:
    lin=lin.strip().split("\t")
    try:
        test_label[lin[0]] = int(lin[1])
    except:
        test_multilabel+=1
        for l in lin[1].split(" "):
            test_label[lin[0]] = lin[1].split(" ")
    
 
print("processing test/sent_test.txt")    
test = []
with open(os.path.join(raw_data_path,'test/sent_test.txt'),'r',
          encoding='utf-8') as f:
    lines = f.readlines()
for lin in lines:
    lin=lin.strip().split("\t")
    id_= lin[0]
    head = lin[1]
    tail = lin[2]
    tokens = lin[3].split(" ")
    head_index = tokens.index(head)
    tail_index = tokens.index(tail)
    if type(test_label.get(id_))==list:
        for l in test_label.get(id_):
            l=int(l)
            test.append({'token':tokens,
                         'h':{'name':head, 'pos':[head_index, head_index+1]},
                         't':{'name':tail, 'pos':[tail_index, tail_index+1]},
                         'label':l,
                         'id':id_})
    else:
        test.append({'token':tokens,
                      'h':{'name':head, 'pos':[head_index, head_index+1]},
                      't':{'name':tail, 'pos':[tail_index, tail_index+1]},
                      'label':test_label.get(id_),
                      'id':id_})
    
print("processing dev/sent_relation_dev.txt")    
dev_label = {}
dev_multilabel=0
with open(os.path.join(raw_data_path,'dev/sent_relation_dev.txt'),'r',
          encoding='utf-8') as f:
    lines = f.readlines()
for lin in lines:
    lin=lin.strip().split("\t")
    try:
        dev_label[lin[0]] = int(lin[1])
    except:
        dev_multilabel+=1
        for l in lin[1].split(" "):
            dev_label[lin[0]] = lin[1].split(" ")
 
print("processing dev/sent_dev.txt")    
dev = []
with open(os.path.join(raw_data_path,'dev/sent_dev.txt'),'r',
          encoding='utf-8') as f:
    lines = f.readlines()
for lin in lines:
    lin=lin.strip().split("\t")
    id_= lin[0]
    head = lin[1]
    tail = lin[2]
    tokens = lin[3].split(" ")
    head_index = tokens.index(head)
    tail_index = tokens.index(tail)
    if type(dev_label.get(id_))==list:
        for l in dev_label.get(id_):
            l=int(l)
            dev.append({'token':tokens,
                         'h':{'name':head, 'pos':[head_index, head_index+1]},
                         't':{'name':tail, 'pos':[tail_index, tail_index+1]},
                         'label':l,
                         'id':id_})
    else:
        dev.append({'token':tokens,
                  'h':{'name':head, 'pos':[head_index, head_index+1]},
                  't':{'name':tail, 'pos':[tail_index, tail_index+1]},
                  'label':dev_label.get(id_),
                  'id':id_})
    
with open(os.path.join(output_data_path,'train.json'),'w',
          encoding='utf-8') as f:
    for item in train:
        json.dump(item, f)
        f.write('\n')
     
with open(os.path.join(output_data_path,'test.json'),'w',
          encoding='utf-8') as f:
    for item in test:
        json.dump(item, f)
        f.write('\n')
     
with open(os.path.join(output_data_path,'dev.json'),'w',
          encoding='utf-8') as f:
    for item in dev:
        json.dump(item, f)
        f.write('\n')
     
print(len(train))
print(train_multilabel)

print(len(test))
print(test_multilabel)

print(len(dev))
print(dev_multilabel)
