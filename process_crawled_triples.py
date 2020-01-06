# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:09:36 2019

@author: admin
"""
import os, json, pickle
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from pyhanlp import * 

NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')

data_path = os.path.join(os.getcwd(),'preprocessed_data')
raw_data_path = os.path.join(os.getcwd(),'raw_data')

if os.path.exists(data_path + '/characters.pkl') and\
   os.path.exists(data_path + '/characters_att.pkl'):
       characters = pickle.load(open(data_path + '/characters.pkl','rb'))
       characters_att = pickle.load(open(data_path + '/characters_att.pkl','rb'))
       potential_entity = pickle.load(open(data_path + '/potential_entity.pkl','rb'))
       characters_copy = pickle.load(open(data_path + '/characters_copy.pkl','rb'))
       fuzzy_names = pickle.load(open(data_path + '/fuzzy_names.pkl','rb'))

else:
    
    entities = set()
    id_entities = set()
    relations = set()
    with open(data_path + '/asoif.2001030836.ttl','r',encoding="utf-8") as f:
        lines = f.readlines()
    for lin in lines[2:]:
        lin=lin.strip().split('\t')
        if lin[0].startswith('e:'): 
            e1 = lin[0][2:].replace('"',"")
        else:
            e1 = lin[0].replace('"',"")
        if lin[2].startswith('e:'): 
            e2 = lin[2][2:].replace('"',"")
        else:
            e2 = lin[2].replace('"',"")
        relations.add(lin[1][2:])
        if 'true_entity:'+e1 not in entities:
            entities.add('true_entity:'+e1)
            id_entities.add(('true_entity:'+e1, len(entities)))
            
        if 'true_entity:'+e2 not in entities:
            entities.add('true_entity:'+e2)
            e2_id = len(entities)
            id_entities.add(('true_entity:'+e2, e2_id))
    
    id_entities_dict = dict(id_entities)        
    with open(data_path + '/candidate_entity_replacement_list_v5.jsonl','r',encoding="utf-8") as f:
        lines = f.readlines()
    f.close()
    for lin in lines:
        #lin = re.sub(' +', '\t', lin) 
        #lin = re.sub('\t+', '\t', lin) 
        lin = HanLP.convertToSimplifiedChinese(lin)
        lin = json.loads(lin)
        
        
        if lin['s'].startswith('e:'): 
            e1 = lin['s'][2:].replace('"',"")
        else:
            e1 = lin['s'].replace('"',"")
            
        if lin['o'].startswith('e:'): 
            e2 = lin['o'][2:].replace('"',"")
        else:
            e2 = lin['o'].replace('"',"")
            
        candidate_entity = [i[2:] for i in lin['candidate_entity']]
        
        if 'true_entity:'+e1 not in entities:
            entities.add('true_entity:'+e1)
            id_entities.add(('true_entity:'+e1, len(entities)))
            
        if 'true_entity:'+e2 not in entities:
            entities.add('true_entity:'+e2)
            e2_id = len(entities)
            id_entities.add(('true_entity:'+e2, e2_id))
        else:
            e2_id = id_entities_dict.get('true_entity:'+e2)
            
        for ce in candidate_entity:
            if ce.startswith('e:'): 
                ce = ce[2:]
            if ce==e2:
                continue
            if 'candidate_entity:'+ce not in entities:
                entities.add('candidate_entity:'+ce)
                id_entities.add(('candidate_entity:'+ce, e2_id))
            
            
        relations.add(lin['r'][2:])

'''    
id_entities = [(i,e) for i,e in enumerate(entities)]
id_entities_ = id_entities.copy()
for id_, cs in id_entities:
    if "·" in cs:
        for i,c in enumerate(cs.split("·")):
            if c not in entities:
                c = c.strip()
                id_entities_.append((id_, c))
                entities.add(c)
    for i,c in enumerate(NLPTokenizer.segment(cs)):
        c_ = str(c).split("/")[0].strip()
        if c not in entities and len(c_)>1 and\
        str(c).split("/")[1].startswith('nr'):
            id_entities_.append((id_, c_))
            entities.add(c_)
'''
with open(data_path +'/literal_vocabulary','w', encoding='utf-8') as f:
    for ent in entities:
        f.write(ent+"\n")
        
id_entities_dict = dict(id_entities)   
with open(data_path +'/ent2id.json','w', encoding='utf-8') as f:
    json.dump(id_entities_dict,f)