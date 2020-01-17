# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:06:21 2020

@author: admin
"""

# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer
#from transformers import *

class Config(object):

    """配置参数"""
    def __init__(self, dataset, num_classes, pad_size,
                 dropout_linear, dropout_other):
        self.model_name = 'bertEntityEncoder'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = dropout_linear                                   # 随机失活
        self.num_classes = num_classes                                  # 类别数
        self.pad_size = pad_size                                        # 每句话处理成的长度(短填长切)
        self.bert_path = './bert_pretrain/bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        
        self.hidden_size = 768
        
        self.feature_size = self.hidden_size


class Model(nn.Module):

    def __init__(self, config,
                 blank_padding=True, mask_entity=False):
        """
        Args:
            max_length: max length of sentence
            pretrain_path: path of pretrain model
        """
        super().__init__()
        self.config = config
        self.restored = False
        self.max_length = config.pad_size
        self.blank_padding = blank_padding
        self.hidden_size = config.hidden_size * 2
        self.mask_entity = mask_entity
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout=nn.Dropout(config.dropout)

    def forward(self, input_):
        """
        Args:
            token: (B, L), index of tokens
            att_mask: (B, L), attention mask (1 for contents and 0 for padding)
            pos1: (B, 1), position of the head entity starter
            pos2: (B, 1), position of the tail entity starter
        Return:
            (B, 2H), representations for sentences
        """
        for index, item in enumerate(input_[-1]):
            token, att_mask, p1, p2 = self.tokenize(item)
            if index==0:
                tokens = token.clone()
                att_masks = att_mask.clone()
                pos1 = p1.clone()
                pos2 = p2.clone()
            else:
                tokens = torch.cat([tokens ,token],axis=0)
                att_masks = torch.cat([att_masks ,att_mask],axis=0)
                pos1 = torch.cat([pos1 ,p1],axis=0)
                pos2 = torch.cat([pos2 ,p2],axis=0)
        
        if torch.cuda.is_available():
            tokens = torch.tensor(tokens).cuda()  # 输入的句子
            att_masks = torch.tensor(att_masks).cuda()  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()
        
        hidden, _ = self.bert(tokens, attention_mask=att_masks)
        # Get entity start hidden state
        onehot = torch.zeros(hidden.size()[:2]).float()  # (B, L)
        if torch.cuda.is_available():
            onehot = onehot.cuda()
            
        onehot_head = onehot.scatter_(1, pos1, 1)
        onehot_tail = onehot.scatter_(1, pos2, 1)
        head_hidden = (onehot_head.unsqueeze(2) * hidden).sum(1)  # (B, H)
        tail_hidden = (onehot_tail.unsqueeze(2) * hidden).sum(1)  # (B, H)
        x = torch.cat([head_hidden, tail_hidden], 1)  # (B, 2H)
        x = self.linear(x)
        x = self.dropout(x)
         
        return x
    

    def tokenize(self, item):
        """
        Args:
            item: data instance containing 'text' / 'token', 'h' and 't'
        Return:
            Name of the relation of the sentence
        """
        # Sentence -> token
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        if not is_token:
            pos_min = pos_head
            pos_max = pos_tail
            if pos_head[0] > pos_tail[0]:
                pos_min = pos_tail
                pos_max = pos_head
                rev = True
            else:
                rev = False
            sent0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            ent0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            sent1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            ent1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
            sent2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            pos_head = [len(sent0), len(sent0) + len(ent0)]
            pos_tail = [
                len(sent0) + len(ent0) + len(sent1),
                len(sent0) + len(ent0) + len(sent1) + len(ent1)
            ]
            if rev:
                pos_tail = [len(sent0), len(sent0) + len(ent0)]
                pos_head = [
                    len(sent0) + len(ent0) + len(sent1),
                    len(sent0) + len(ent0) + len(sent1) + len(ent1)
                ]
            tokens = sent0 + ent0 + sent1 + ent1 + sent2
        else:
            tokens = sentence

        # Token -> index
        re_tokens = ['[CLS]']
        cur_pos = 0
        pos1 = 0
        pos2 = 0
        for token in tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                pos1 = len(re_tokens)
                re_tokens.append('[unused0]')
            if cur_pos == pos_tail[0]:
                pos2 = len(re_tokens)
                re_tokens.append('[unused1]')
            re_tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[1] - 1:
                re_tokens.append('[unused2]')
            if cur_pos == pos_tail[1] - 1:
                re_tokens.append('[unused3]')
            cur_pos += 1
        re_tokens.append('[SEP]')
        pos1 = min(self.max_length - 1, pos1)
        pos2 = min(self.max_length - 1, pos2)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(re_tokens)
        avai_len = len(indexed_tokens)

        # Position
        pos1 = torch.tensor([[pos1]]).long()
        pos2 = torch.tensor([[pos2]]).long()

        # Padding
        if self.blank_padding:
            while len(indexed_tokens) < self.max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[:self.max_length]
        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(
            0)  # (1, L)

        # Attention mask
        att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
        att_mask[0, :avai_len] = 1

        return indexed_tokens, att_mask, pos1, pos2


class Classifier(nn.Module):
    """classifier model for relation extraction."""

    def __init__(self, config):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(config.hidden_size * 2, 300)
        self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(300, config.num_classes)
        
    def forward(self, feat):
        """Forward the classifier."""
        out = self.fc(feat)
        out = self.dropout2(out)
        out = self.fc2(out)
        return out
