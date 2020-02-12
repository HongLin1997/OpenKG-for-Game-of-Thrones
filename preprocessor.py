# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 22:19:21 2019

@author: admin
"""
import os,re, json, datetime, pickle
from html.parser import HTMLParser
from pyhanlp import *

NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
CustomDictionary = JClass("com.hankcs.hanlp.dictionary.CustomDictionary")
    
class MyHTMLParser(HTMLParser):
    text = []
    family = []
    family_members = []
    chapter = ''
    previous_data = None
    id_ = 0
    text_add = False
    def handle_starttag(self, tag, attrs):
        if 'p' in tag or 'h2' in tag:
            self.text_add = True 
        pass
    
    def handle_endtag(self, tag):
        if 'h2' in tag:
            self.chapter = self.text[-1][2]
        
        #print("Encountered an end tag :", tag)

    def handle_data(self, data):
        if len(data.strip())>0 and self.text_add:
            data = re.sub('<(.*?)>', "", data.strip())
            self.text.append((self.id_, self.chapter, data))
            self.id_ += 1
            self.text_add = False 
            
        if len(data.strip())>0 and \
           data.endswith('家族'):
               self.family.append(data.strip())
               
        if data.strip().startswith('——') and len(data.strip())>3 and \
           len(self.family)>0:
               if len(self.family_members)==0 or \
                  self.family_members[-1][0] != self.family[-1]:
                      # new family
                      self.family_members.append((self.family[-1], self.previous_data.strip()))
                      
               self.family_members.append((self.family[-1], data.strip()))
               
        if len(data.strip())>0:
            self.previous_data = data
        
        return data


#- body:
class Preprocessor():
    def __init__(self, parser, sentence_min_length):
        self.time = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H:%M:%S")
        self.parser = parser
        self.sentence_min_length = sentence_min_length
    # 把文章的content变为句子list
    def tokenize(self):
        # load the filtering list:
        if os.path.exists("filter_default"):
            with open("filter_default",'r',encoding='utf-8') as f:
                filters = f.read().splitlines()
            f.close()
        else:
            filters=[]

        # split the whole passage into the sentences
        # objects are saved in the .textObj:
        tok = []
        for item in self.textObj:
            try:
                para = item['content']
                para = re.sub('\u3000', "", para)  # 空格
                para = re.sub('\xa0', "", para)  # 空格
                para = re.sub(' ', "", para)  # 空格
                #para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
                #para = re.sub(';', r"\1\n\2", para)  # 单字符断句分号
                #para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
                #para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
                #para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
                # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
                para = para.rstrip()  # 段尾如果有多余的\n就去掉它
                # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
                res = []
                for x in re.split("\n|；",para):
                    x = x.strip()
                    if x in res:
                        continue
                    if len(x)>=self.sentence_min_length and \
                       x not in filters and x not in res: # the threshold is 20 words in total.
                        res.append(x)
                #会打乱顺序
                #res = list(set(res))
                item['content'] = res
                if len(res)>0:
                    tok += [item]
                else:
                    next
            except KeyError as e:
                print('SolrError: key-<%s> of <%s> is missing. '%(e,item['id'])) # skip this error (no valid text is pumped into the pipe)
                next

        #print("Process: %s records are tokenized."%(len(tok)))
        self.tokenObj = tok
    
    def EnumPathFiles(self, path):
        if not os.path.isdir(path):
            print('Error:"',path,'" is not a directory or does not exist.')
            return
        list_dirs = os.walk(path)
        path_file=[]
        for root, dirs, files in list_dirs:
            for file in files:
                if os.path.splitext(file)[1] == '.xhtml':
                    path_file.append({"title":file, "path":os.path.join(root,file)})
        return path_file
    
    def inputData(self,INPUT=None, INPUT_PATH=None):
        # make sure the input cotains only textual paragraphs:
        # for json format data, please decode it before inputting:
        if INPUT_PATH!=None:
            path_file = self.EnumPathFiles(INPUT_PATH)
            data=[]         
            for item in path_file:
                with open(item.get("path"),'r',encoding="utf-8") as f:
                    data_ = f.read()
                f.close()
                self.parser.feed(data_)
                for i in self.parser.text:
                    data.append({'content':i[2],'chapter':i[1],
                                 'title':item.get("title")[0:-6],
                                 'id':i[0]})
            
                #data_= "".join([i[1] for i in self.parser.text])
                self.parser.text = []
                self.parser.id_ = 0
                self.parser.chapter = ""
                #data.append({'content':data_,'title':item.get("title")[0:-6],'id':'none'} )
            self.textObj = data
            
            return
    
        if isinstance(INPUT,str):
            if os.path.isfile(INPUT):
                with open(INPUT,'r') as f:
                    data = f.read().splitlines()
                f.close()
                # - different types:
                if INPUT.endswith('.json') or INPUT.endswith('.jsonl'):
                    
                    data = [eval(re.sub('NaN','"none"',x)) for x in data]
                    for i in range(len(data)):
                        data[i]['id'] = 'none'
                        
                elif INPUT.endswith('.txt'):
                    data = [{'content':x,'title':'none','id':'none'} for x in data]
                    
                else:
                    data = None
                    print("InputError: can only parse .json or .txt")
                
            else:
                data = None
                print("InputError: wrong directory.")
                
        elif isinstance(INPUT,list):
            data = [x for x in INPUT if len(x)>20] # default threshold:

        else:
            data = None
            print("InputError: wrong directory.")
        
        self.textObj = data

    
    def wordFilter(self,mode='keyword',*words):
        if len(words) < 1: # no specific keywords loaded
            # by default, use the local keywords list:
            if os.path.exists('keywords_default'):
                with open('keywords_default','r',encoding='utf-8') as f:
                    kws = f.read().splitlines()
                f.close()
            else:
                kws =[]

        else:
            # input: keywords list:
            kws = [x for x in words] # serialize the words
        # pre-filtering by [filter_log]:
        if "filter_log" in os.listdir():
            with open("filter_log","r") as f:
                flog = f.read().splitlines()
                flog = [x.split(">>")[-1] for x in flog]
            f.close()
        else:
            flog = [""]

        # filter the logged records:
        # two ways of filtering the sentences that we need:
        tokens = [x['content'] for x in self.tokenObj] # serialize 
        snippets = []
        # - keywords stats:
        if mode == "keyword":
            # len(tokens)： the length of all passage
            for i in range(0,len(tokens)):
                tks = tokens[i]
                # tks: [sentence1, sentence2,...]
                tks = list(set(tks)) # simply remove the "totally alike" replicates
                count = 0 #the number of keywords in this passages
                #tk = []
                for t in tks:
                    for k in kws:
                        if k in t:
                            count+=1
                            #tk += [t]
                        else:
                            next
                 
                print("Process: [%s] in stack."%(self.tokenObj[i]['id']))
                snippets += [{"id":self.tokenObj[i]['id'],
                              "title":self.tokenObj[i]['title'],
                              "chapter":self.tokenObj[i]['chapter'],
                              "snip":re.sub('[\r\n\t\xa0]','',x),
                              'hanlp_tokens': [str(i).split("/")[0] for i in list(NLPTokenizer.segment(re.sub('[\r\n\t\xa0]','',x)))],
                              'hanlp_pos': [str(i).split("/")[1] for i in list(NLPTokenizer.segment(re.sub('[\r\n\t\xa0]','',x)))]} for x in tks ]
                
                
        else:
            pass # remained: !!!!!

        # save in tokens
        print("Process: %s tokens are filtered."%(len(snippets)))
        self.tokens = snippets

    def tokenAsAnnotation(self,path=None,name="sample.txt"):
        if path is None:
            path = os.getcwd()
            if name not in os.listdir():
                f = open(name,'w', encoding="utf-8")
                f.close()

            with open(path+name,'a') as f:
                for item in self.tokens:
                    # convert to .jsonl for prodigy's annotation:
                    if name.endswith('.json') or name.endswith('.jsonl'):
                        sn = json.dumps({"text":item['snip'],
                                         "meta":{"source":item['title'],
                                                 "chapter":item['chapter'],
                                                 'id': item['id']}})
                        f.write(sn + "\n")
                    else:
                        sn = item['snip']
                        f.write(sn + "\n")
                        
            f.close()                

        else:
            if name not in os.listdir(path):
                f = open(name,'w', encoding="utf-8")
                f.close()
            
            with open(os.path.join(path, name),'w', encoding="utf-8") as f:
                for item in self.tokens:
                    if name.endswith('.json') or name.endswith('.jsonl'):
                        sn = json.dumps({"text":item['snip'], 
                                         'hanlp_tokens': item['hanlp_tokens'],
                                         'hanlp_pos': item['hanlp_pos'],
                                         "meta":{"source":item['title'],
                                                 "chapter":item['chapter'],
                                                 'id': item['id']}})
                        f.write(sn + "\n")
                    else:
                        sn = item['snip']
                        f.write(sn + "\n")

            f.close()
              
        # -----
        # print("""Process: annotation tokens are ready, \n \t see <%s> in [%s]."""%(name,path))

def main():   
    os.chdir(os.getcwd())
    data_path = os.path.join(os.getcwd(),'raw_data')
    output_path = os.path.join(os.getcwd(),'preprocessed_data')

    #characters = pickle.load(open(output_path + '/characters.pkl','rb'))
    
    with open(output_path +'/literal_vocabulary','r', encoding='utf-8') as f:
        lines = f.readlines()
        characters = [i.strip().split(":")[1] for i in lines]
        tag = [i.strip().split(":")[0] for i in lines]
    #characters_att = pickle.load(open(output_path + '/characters_att.pkl','rb'))
    for t, cs in zip(tag, characters):
        print(cs, t)
        if t=='true_entity':
            CustomDictionary.insert(cs, t+" 2048")
        else:
            CustomDictionary.insert(cs, t+" 1024")
    with open(data_path + '/stopwords/哈工大停用词表.txt','r',
          encoding="utf-8") as f:
        stopwords = [s.strip() for s in f.readlines()]

    stopwords.append('…')
    for s in stopwords:
        CustomDictionary.insert(s, "stopwords 1024")
        
    if os.path.exists(output_path)==False:
        os.mkdir(output_path)
        
    parser = MyHTMLParser()

    sentence_min_length = 5
    proc = Preprocessor(parser, sentence_min_length) # solr starts by default:
    
    proc.inputData(INPUT_PATH=data_path)
    proc.tokenize()
    proc.wordFilter()
    proc.tokenAsAnnotation(path=output_path,
                           name="preprocessed_data.jsonl")

if __name__ == "__main__":
    main()