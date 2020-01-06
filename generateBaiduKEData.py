# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 19:58:34 2020

@author: admin
"""
import os,json
import pandas as pd
import random

raw_data_path= os.path.join(os.getcwd(),'raw_data','baidu_knowledge_extraction')
out_path= os.path.join(os.getcwd(),'deepke-master','data','origin')

with open(raw_data_path + '/train_data.json','r',encoding="utf-8") as f:
    lines = f.readlines()
train_df=[]    
for lin in lines:
    lin = json.loads(lin)
    sentence = lin['text'].lower()
    tokens = [i['word'] for i in lin['postag']]
    for temp in lin['spo_list']:
        relation = temp['predicate']
        head = temp['subject'].lower()
        tail = temp['object'].lower()
        head_offset = sentence.index(head) if head in sentence else -1
        head_idx = tokens.index(head) if head in tokens else -1
        if head_idx!=-1:
            tokens[head_idx]=' head '
        head_type = temp['subject_type']
        tail_offset = sentence.index(tail) if tail in sentence else -1
        tail_idx = tokens.index(tail) if tail in tokens else -1
        if tail_idx!=-1:
            tokens[tail_idx]=' tail '
        tail_type = temp['object_type']
        
        train_df += [{'sentence':sentence, 'tokens':tokens, 'relation': relation,
                     'head':head, 'head_offset':head_offset,
                     'head_type': head_type, 'head_idx':head_idx,
                     'tail':tail, 'tail_offset':tail_offset, 
                     'tail_type': tail_type, 'tail_idx': tail_idx}]
    
train_df = pd.DataFrame(train_df)
print(len(train_df))
train_df = train_df[(train_df['head_offset']!=-1) & (train_df['tail_offset']!=-1) &\
                    (train_df['head_idx']!=-1) & (train_df['tail_idx']!=-1)]
print(len(train_df))
train_df.to_csv(out_path+'/train.csv',index=False)

with open(raw_data_path + '/dev_data.json','r',encoding="utf-8") as f:
    lines = f.readlines()
    
random.shuffle(lines)
valid_df=[]    
for lin in lines[0:int(len(lines)/2)]:
    lin = json.loads(lin)
    sentence = lin['text'].lower()
    tokens = [i['word'] for i in lin['postag']]
    for temp in lin['spo_list']:
        relation = temp['predicate']
        head = temp['subject'].lower()
        tail = temp['object'].lower()
        head_offset = sentence.index(head) if head in sentence else -1
        head_idx = tokens.index(head) if head in tokens else -1
        if head_idx!=-1:
            tokens[head_idx]=' head '
        head_type = temp['subject_type']
        tail_offset = sentence.index(tail) if tail in sentence else -1
        tail_idx = tokens.index(tail) if tail in tokens else -1
        if tail_idx!=-1:
            tokens[tail_idx]=' tail '
        tail_type = temp['object_type']
        
        valid_df += [{'sentence':sentence, 'tokens':tokens, 'relation': relation,
                     'head':head, 'head_offset':head_offset,
                     'head_type': head_type, 'head_idx':head_idx,
                     'tail':tail, 'tail_offset':tail_offset, 
                     'tail_type': tail_type, 'tail_idx': tail_idx}]
valid_df = pd.DataFrame(valid_df)
print(len(valid_df))
valid_df = valid_df[(valid_df['head_offset']!=-1) & (valid_df['tail_offset']!=-1) &\
                    (valid_df['head_idx']!=-1) & (valid_df['tail_idx']!=-1)]
print(len(valid_df))
valid_df.to_csv(out_path+'/valid.csv',index=False)

test_df=[]
for lin in lines[int(len(lines)/2):]:
    lin = json.loads(lin)
    sentence = lin['text'].lower()
    tokens = [i['word'] for i in lin['postag']]
    for temp in lin['spo_list']:
        relation = temp['predicate']
        head = temp['subject'].lower()
        tail = temp['object'].lower()
        head_offset = sentence.index(head) if head in sentence else -1
        head_idx = tokens.index(head) if head in tokens else -1
        if head_idx!=-1:
            tokens[head_idx]=' head '
        head_type = temp['subject_type']
        tail_offset = sentence.index(tail) if tail in sentence else -1
        tail_idx = tokens.index(tail) if tail in tokens else -1
        if tail_idx!=-1:
            tokens[tail_idx]=' tail '
        tail_type = temp['object_type']
        
        test_df += [{'sentence':sentence, 'tokens':tokens, 'relation': relation,
                     'head':head, 'head_offset':head_offset,
                     'head_type': head_type, 'head_idx':head_idx,
                     'tail':tail, 'tail_offset':tail_offset, 
                     'tail_type': tail_type, 'tail_idx': tail_idx}]
test_df = pd.DataFrame(test_df)
print(len(test_df))
test_df = test_df[(test_df['head_offset']!=-1) & (test_df['tail_offset']!=-1) &\
                    (test_df['head_idx']!=-1) & (test_df['tail_idx']!=-1)]
print(len(test_df))
test_df.to_csv(out_path+'/test.csv',index=False)


with open(raw_data_path + '/all_50_schemas','r',encoding="utf-8") as f:
    lines = f.readlines()
    
relations=[{'head_type':'None', 
            'tail_type':'None', 
            'relation':'None',
            'index':0}]
for lin in lines:
    lin = json.loads(lin)
    relations+=[{'head_type':lin['subject_type'],
                 'tail_type':lin['object_type'],
                 'relation':lin['predicate'],
                 'index':len(relations)}]
relations = pd.DataFrame(relations)
relations.to_csv(out_path+'/relation.csv',index=False)
