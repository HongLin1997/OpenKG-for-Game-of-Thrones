# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:09:36 2019

@author: admin
"""
import os, json, pickle
from pyhanlp import * 
from itertools import combinations
import random

data_path = os.path.join(os.getcwd(),'preprocessed_data')
output_path = os.path.join(os.getcwd(),'datasets/GOT')
raw_data_path = os.path.join(os.getcwd(),'raw_data')

with open(raw_data_path + '/stopwords/哈工大停用词表.txt','r',
          encoding="utf-8") as f:
    stopwords = [s.strip() for s in f.readlines()]
stopwords = stopwords[0:263]
stopwords.append('…')

# 用dict的方式存储三元组: {(e1,e2):[relation, relation,...], ...}
true_triples = {}
potential_triples = {}
with open(data_path + '/asoif.2001030836.ttl', 'r', encoding="utf-8") as f:
    lines = f.readlines()
for lin in lines[2:]:
    lin = HanLP.convertToSimplifiedChinese(lin)
    lin = lin.strip().split('\t')
    if lin[0].startswith('e:'):
        e1 = lin[0][2:].replace('"', "")
    else:
        e1 = lin[0].replace('"', "")
    if lin[2].startswith('e:'):
        e2 = lin[2][2:].replace('"', "")
    else:
        e2 = lin[2].replace('"', "")

    true_triples[(e1, e2)] = true_triples.get((e1, e2), []) + [lin[1][2:].strip()]

# true_triples记录ttl中存在的正确的三元组
# potential_triples记录根据candidate entity得到的可能的三元组
with open(data_path + '/candidate_entity_replacement_list_v5.jsonl', 'r', encoding="utf-8") as f:
    lines = f.readlines()

f.close()
for lin in lines:
    
    lin = HanLP.convertToSimplifiedChinese(lin)
    if "盖瑞" in lin:
        print(lin)
    lin = json.loads(lin)
    
    if lin['s'].startswith('e:'):
        e1 = lin['s'][2:].replace('"', "")
    else:
        e1 = lin['s'].replace('"', "")

    if lin['o'].startswith('e:'):
        e2 = lin['o'][2:].replace('"', "")
    else:
        e2 = lin['o'].replace('"', "")

    if e1 == '' or e2 == '':
        continue

    true_triples[(e1, e2)] = true_triples.get((e1, e2), []) + [lin['r'][2:].strip()]

    for ce in lin['candidate_entity']:
        if ce.startswith('e:'):
            ce = ce[2:]
        if ce == e2:
            continue
        potential_triples[(e1, ce)] = potential_triples.get((e1, ce), []) + [lin['r'][2:].strip()]

# 到句子里寻找能够匹配三元组的句子
with open(data_path + '/preprocessed_data.jsonl', 'r',
          encoding="utf-8") as f:
    lines = f.readlines()
    
negative_corpus = []
only_for_test = []
all_nrs = set()
positive_count = {}
relatin2id={}
for index, lin in enumerate(lines):
    if index % 10000 == 0:
        print('processing progress:', index, '/', len(lines))

    temp = json.loads(lin.strip())

    character_flag = []
    filtered_tokens = []
    filtered_pos = []
    position = -1

    nrs_in_sentence = set()
    for i, (tk, pos) in enumerate(zip(temp['hanlp_tokens'],
                                      temp['hanlp_pos'])):
        # if tk in stopwords:
        #     continue
        filtered_tokens.append(tk)
        filtered_pos.append(pos)
        position += 1
        if tk!='无' and ((pos.startswith('nr') and len(tk) > 1) or
        pos == 'true_entity' or pos == 'candidate_entity'):
            nrs_in_sentence.add((tk, pos, position))
            all_nrs.add((tk, pos))

    temp['token'] = filtered_tokens
    temp['filtered_pos'] = filtered_pos

    entity_pairs = list(combinations(nrs_in_sentence, 2))
    for ent in entity_pairs:
        ((e1, e1_tag, e1_pos), (e2, e2_tag, e2_pos)) = ent
        if e1==e2:
            continue
        relation = None
        if (e1, e2) in true_triples.keys():
            for r in true_triples.get((e1, e2)):
                relation = r + '(e1,e2)'
                temp = temp.copy()
                temp['ents'] = ent
                if relation not in relatin2id.keys():
                    relatin2id[relation] = len(relatin2id)
                temp['label'] = relatin2id[relation]
                temp['h']={'name':e1, 'pos':[e1_pos, e1_pos+1], 'tag':e1_tag}
                temp['t']={'name':e2, 'pos':[e2_pos, e2_pos+1], 'tag':e2_tag}
                
                if e1_tag.startswith('nr') or e2_tag.startswith('nr'):
                    if temp not in only_for_test:
                        only_for_test += [temp]
                else:    
                    if relation not in positive_count.keys():
                        #corpus += [temp]
                        positive_count[relation] = positive_count.get(relation, []) + [temp]
                    elif temp not in positive_count.get(relation):
                        positive_count[relation] = positive_count.get(relation, []) + [temp]
                    
        # === cxy code: elif -> if===
        if (e2, e1) in true_triples.keys():
            for r in true_triples.get((e2, e1)):
                relation = r + '(e2,e1)'
                temp = temp.copy()
                temp['ents'] = ent
                if relation not in relatin2id.keys():
                    relatin2id[relation] = len(relatin2id)
                temp['label'] = relatin2id[relation]
                temp['h']={'name':e2, 'pos':[e2_pos, e2_pos+1], 'tag':e2_tag}
                temp['t']={'name':e1, 'pos':[e1_pos, e1_pos+1], 'tag':e1_tag}
                if e1_tag.startswith('nr') or e2_tag.startswith('nr'):
                    if temp not in only_for_test:
                        only_for_test += [temp]
                else:    
                    if relation not in positive_count.keys():
                        #corpus += [temp]
                        positive_count[relation] = positive_count.get(relation, []) + [temp]
                    elif temp not in positive_count.get(relation):
                        positive_count[relation] = positive_count.get(relation, []) + [temp]
                    
        # === cxy code: ===
        if relation:
            continue
        # =================

        # when we cannot find the true relationA
        # === cxy code: elif -> if===
        if (e1, e2) in potential_triples.keys():
            for r in potential_triples.get((e1, e2)):
                relation = r + '(e1,e2)'
                temp = temp.copy()
                temp['ents'] = ent
                if relation not in relatin2id.keys():
                    relatin2id[relation] = len(relatin2id)
                temp['label'] = relatin2id[relation]
                temp['h']={'name':e1, 'pos':[e1_pos, e1_pos+1], 'tag':e1_tag}
                temp['t']={'name':e2, 'pos':[e2_pos, e2_pos+1], 'tag':e2_tag}
                if e1_tag.startswith('nr') or e2_tag.startswith('nr'):
                    if temp not in only_for_test:
                        only_for_test += [temp]
                else:    
                    if relation not in positive_count.keys():
                        #corpus += [temp]
                        positive_count[relation] = positive_count.get(relation, []) + [temp]
                    elif temp not in positive_count.get(relation):
                        positive_count[relation] = positive_count.get(relation, []) + [temp]
                    
        # === cxy code: elif -> if===
        if (e2, e1) in potential_triples.keys():
            for r in potential_triples.get((e2, e1)):
                relation = r + '(e2,e1)'
                temp = temp.copy()
                temp['ents'] = ent
                if relation not in relatin2id.keys():
                    relatin2id[relation] = len(relatin2id)
                temp['label'] = relatin2id[relation]
                temp['h']={'name':e2, 'pos':[e2_pos, e2_pos+1], 'tag':e2_tag}
                temp['t']={'name':e1, 'pos':[e1_pos, e1_pos+1], 'tag':e1_tag}
                if e1_tag.startswith('nr') or e2_tag.startswith('nr'):
                    if temp not in only_for_test:
                        only_for_test += [temp]
                else:    
                    if relation not in positive_count.keys():
                        #corpus += [temp]
                        positive_count[relation] = positive_count.get(relation, []) + [temp]
                    elif temp not in positive_count.get(relation):
                        positive_count[relation] = positive_count.get(relation, []) + [temp]
                    
        else:
            temp = temp.copy()
            temp['ents'] = ent
            if relation not in relatin2id.keys():
                relatin2id[relation] = len(relatin2id)
            temp['label'] = relatin2id[relation]
            temp['h']={'name':e1, 'pos':[e1_pos, e1_pos+1], 'tag':e1_tag}
            temp['t']={'name':e2, 'pos':[e2_pos, e2_pos+1], 'tag':e2_tag}
            if e1_tag.startswith('nr') or e2_tag.startswith('nr'):
                if temp not in only_for_test:
                    only_for_test += [temp]
            else:    
                if temp not in negative_corpus:
                    negative_corpus += [temp]
                    #corpus += [temp]

positive_corpus_train = []
positive_corpus_dev = []
positive_corpus_eval = []

for c in positive_count.values():
    random.shuffle(c)
    positive_corpus_train += c[0:int(len(c)*0.7)]
    positive_corpus_dev += c[int(len(c)*0.7):int(len(c)*0.8)]
    positive_corpus_eval += c[int(len(c)*0.8):]

random.shuffle(negative_corpus)
random.shuffle(only_for_test)
corpus_train = positive_corpus_train + \
               negative_corpus[0:int(len(negative_corpus)*0.7)]
corpus_dev = positive_corpus_dev + \
             negative_corpus[int(len(negative_corpus)*0.7):int(len(negative_corpus)*0.8)]
corpus_eval = positive_corpus_eval  + \
              negative_corpus[int(len(negative_corpus)*0.8):] + \
              only_for_test


corpus = list(corpus_train) + list(corpus_dev) + list(corpus_eval)
print(len(corpus))
print(len(only_for_test))
print([(k,len(v)) for k,v in positive_count.items()])


with open(output_path + '/corpus_train.jsonl', 'w', encoding='utf-8') as f:
    for c in corpus_train:
        json.dump(c, f)
        f.write("\n")

with open(output_path + '/corpus_dev.jsonl', 'w', encoding='utf-8') as f:
    for c in corpus_dev:
        json.dump(c, f)
        f.write("\n")

with open(output_path + '/corpus_eval.jsonl', 'w', encoding='utf-8') as f:
    for c in corpus_eval:
        json.dump(c, f)
        f.write("\n")

with open(output_path + '/tagged_corpus.jsonl', 'w', encoding='utf-8') as f:
    for c in corpus:
        json.dump(c, f)
        f.write("\n")

with open(output_path + '/relatin2id.jsonl', 'w', encoding='utf-8') as f:
    json.dump(relatin2id, f)
        
'''
with open(data_path + '/supervise_data_v2.jsonl','r',encoding="utf-8") as f:
    lines = f.readlines()     
supervise_data_v2=[]
for lin in lines:
    temp =json.loads(lin)
    if temp not in supervise_data_v2:
        supervise_data_v2+=[temp]
'''