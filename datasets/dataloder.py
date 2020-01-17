# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 21:10:27 2020

@author: admin
"""

from torch.utils.data import Dataset
import json

class CCKSData(Dataset):

    def __init__(self, root_path):
        print('loading %s data'%str(root_path))
        with open(root_path,'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.x = []
        self.labels = []
        
        for lin in lines:
            lin=json.loads(lin)
            
            self.labels.append(int(lin['label']))
            self.x.append(lin)
            
        self.x = list(zip(self.x, self.labels))
            
        print('number of samples: ',len(self.labels))
        print('loading finish')

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)
    
    
class GOTData(Dataset):

    def __init__(self, root_path):
        print('loading %s data'%str(root_path))
        with open(root_path,'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.x = []
        self.labels = []
        
        for lin in lines:
            lin=json.loads(lin)
            self.labels.append(int(lin['label']))
            self.x.append(lin)
            
        self.x = list(zip(self.x, self.labels))
            
        print('number of samples: ',len(self.labels))
        print('loading finish')

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)