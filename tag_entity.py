# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:09:36 2019

@author: admin
"""
import os, json, pickle
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from pyhanlp import *
from itertools import combinations

data_path = os.path.join(os.getcwd(), 'run_LH_code')
raw_data_path = os.path.join(os.getcwd(), 'raw_data')

# stopwords为一个存储停用词的list
with open(raw_data_path + '/stopwords/哈工大停用词表.txt', 'r',
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
corpus = []
all_nrs = set()
positive_count = {}
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
        if ((pos.startswith('nr') and len(tk) > 1) or
                pos == 'true_entity' or pos == 'candidate_entity'):
            nrs_in_sentence.add((tk, pos, position))
            all_nrs.add((tk, pos))

    temp['filtered_tokens'] = filtered_tokens
    temp['filtered_pos'] = filtered_pos

    entity_pairs = list(combinations(nrs_in_sentence, 2))
    for ent in entity_pairs:
        ((e1, _, _), (e2, _, _)) = ent
        relation = None
        if (e1, e2) in true_triples.keys():
            for r in true_triples.get((e1, e2)):
                relation = r + '(e1,e2)'
                temp = temp.copy()
                temp['ents'] = ent
                temp['relation'] = relation
                if temp not in corpus:
                    corpus += [temp]
                    positive_count[relation] = positive_count.get(relation, 0) + 1

        # === cxy code: elif -> if===
        if (e2, e1) in true_triples.keys():
            for r in true_triples.get((e2, e1)):
                relation = r + '(e2,e1)'
                temp = temp.copy()
                temp['ents'] = ent
                temp['relation'] = relation
                if temp not in corpus:
                    corpus += [temp]
                    positive_count[relation] = positive_count.get(relation, 0) + 1

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
                temp['relation'] = relation
                if temp not in corpus:
                    corpus += [temp]
                    positive_count[relation] = positive_count.get(relation, 0) + 1

        # === cxy code: elif -> if===
        if (e2, e1) in potential_triples.keys():
            for r in potential_triples.get((e2, e1)):
                relation = r + '(e2,e1)'
                temp = temp.copy()
                temp['ents'] = ent
                temp['relation'] = relation
                if temp not in corpus:
                    corpus += [temp]
                    positive_count[relation] = positive_count.get(relation, 0) + 1

        else:
            temp = temp.copy()
            temp['ents'] = ent
            temp['relation'] = relation
            if temp not in corpus:
                corpus += [temp]

print(len(corpus))
print(positive_count)

with open(data_path + '/tagged_corpus.jsonl', 'w', encoding='utf-8') as f:
    for c in corpus:
        json.dump(c, f)
        f.write("\n")

'''
with open(data_path + '/supervise_data_v2.jsonl','r',encoding="utf-8") as f:
    lines = f.readlines()     
supervise_data_v2=[]
for lin in lines:
    temp =json.loads(lin)
    if temp not in supervise_data_v2:
        supervise_data_v2+=[temp]
'''
