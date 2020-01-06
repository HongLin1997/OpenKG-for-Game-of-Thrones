from SPARQLWrapper import SPARQLWrapper, JSON
import pickle
import pyhanlp
import json
from triple_crawler import squeeze_result


def sentence_generator(filename):
    with open(filename) as f:
        while True:
            line = f.readline()
            if line != "":
                yield json.loads(line)
            else:
                return ""


def list_index_all(l, v):
    """
    返回列表l中所有值v的索引
    :param l:
    :param v:
    :return:
    """
    ret = []
    while True:
        try:
            ret.append(l.index(v, 0 if len(ret) == 0 else ret[-1] + 1))
        except ValueError:
            return ret


def match_entity(words):
    """
    找出word可能指代的实体，当word与实体的name、名、别名匹配时，这个实体可能就是word指代的对象
    :param words:
    :return:
    """
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        f"""
            PREFIX	r:	<http://kg.course/action/>
            PREFIX	e:	<http://kg.course/entity/>
            SELECT DISTINCT ?s
            WHERE{{
                {{?s r:name "{words}"}}
                UNION{{?s r:别名 "{words}"}}
                UNION{{?s r:名 "{words}"}}
            }}
        """
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    return [squeeze_result(result["s"]) for result in results]


def add_entity2sentence(in_filename, out_filename):
    """
    将匹配到的实体添加到句子的json结构中去
    :param in_filename:
    :param out_filename:
    :return: 返回匹配到的实体集合
    """
    matched_entity = set()
    with open(out_filename, mode="a", encoding="utf-8", errors="ignore") as outfile:
        for sentence in sentence_generator(in_filename):
            print(sentence["text"])
            sentence["entity"] = {}
            # 章节pov人物对应的实体
            entities = match_entity(sentence["meta"]["chapter"])
            matched_entity.update(entities)
            if len(entities):
                sentence["entity"]["pov"] = entities
            for i, w in enumerate(sentence["hanlp_tokens"]):
                # 句子中每个词语寻找对应的实体
                entities = match_entity(w)
                matched_entity.update(entities)
                if len(entities):
                    sentence["entity"][i] = entities
            outfile.write(json.dumps(sentence) + "\n")
    return matched_entity


def unmatched_entity(matched_entity):
    """
    在全部实体的集合中减去matched_entity中的实体
    :param matched_entity:
    :return:
    """
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        f"""
               PREFIX	r:	<http://kg.course/action/>
               PREFIX	e:	<http://kg.course/entity/>
               SELECT DISTINCT ?s
               WHERE{{
                    ?s r:name ?o
               }}
           """
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    all_entity = set([squeeze_result(r["s"]) for r in results])
    return all_entity.difference(matched_entity)


def find_entity_in_sentence(entity):
    """
    找出含有entity的句子
    :param entity:
    :return: list of dict, 一个dict为一个句子
    """
    ret = []
    for s in sentence_generator("processed_data/preprocessed_data_with_entity.jsonl"):
        for _, v in s["entity"].items():
            if entity in v:
                ret.append(s)
                break
    return ret


def find_word_in_sentence(word):
    """
    找出含有word的句子
    :param word:
    :return: list of dict, 一个dict为一个句子
    """
    ret = []
    for s in sentence_generator("processed_data/preprocessed_data_with_entity.jsonl"):
        if word in s["hanlp_tokens"]:
            ret.append(s)
    return ret


def avg_num_of_entity_per_word():
    """
    统计每个词语匹配到的实体数的平均值
    :return: 全部实体数(被重复匹配的实体重复计数)/有匹配实体的词语数
    """
    num_word = 0
    num_entity = 0
    for s in sentence_generator("processed_data/preprocessed_data_with_entity.jsonl"):
        num_word += len(s["entity"])
        for _, v in s["entity"].items():
            num_entity += len(v)
    return num_entity/num_word


def one_entity_sentence():
    """
    找出只有一个实体的句子
    :return:
    """
    ret = []
    for s in sentence_generator("processed_data/preprocessed_data_with_entity.jsonl"):
        if len(s["entity"]) <= 1:
            ret.append(s)
    return ret


def entity2sentence(in_filename, out_filename):
    ret = dict()
    with open(in_filename) as o_f:
        for i, l in enumerate(o_f.readlines()):
            s = json.loads(l)
            for offset, entity_list in s["entity"].items():
                for e in entity_list:
                    if e not in ret.keys():
                        ret[e] = {(i, offset)}
                    else:
                        ret[e].add((i, offset))
    with open(out_filename, mode="wb") as i_f:
        pickle.dump(ret, i_f)


if __name__ == "__main__":
    # for s in sentence_generator("processed_data/preprocessed_data.jsonl"):
    #     break
    # sentence = "徐先生还具体帮助他确定了把画雄鹰、松鼠和麻雀作为主攻目标。"
    # sentence = "我新造一个词叫幻想乡你能识别并正确标注词性吗？"
    # NLPTokenizer = pyhanlp.JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
    # a = NLPTokenizer.segment(sentence)
    # b = NLPTokenizer.analyze(sentence)
    # c = NLPTokenizer.analyze(sentence).translateLabels()
    # inFilename = "processed_data/preprocessed_data.jsonl"
    # outFilename = "processed_data/preprocessed_data_with_entity_v1.jsonl"
    # matchedEntity = add_entity2sentence(inFilename, outFilename)
    # unmatchedEntity = unmatched_entity(matchedEntity)
    # with open("match", mode="a", encoding="utf-8") as f:
    #     for e in matchedEntity:
    #         f.write(e+"\n")
    # with open("unmatch", mode="a", encoding="utf-8") as f:
    #     for e in unmatchedEntity:
    #         f.write(e+"\n")
    # s_list = find_entity_in_sentence("e:热派")
    # w_list = find_word_in_sentence("加尔斯")
    # num_sentence = 0
    # num_matched_word = [len(s["entity"])
    #                     for s in sentence_generator("processed_data/preprocessed_data_with_entity_v1.jsonl")]
    entity2sentence("processed_data/preprocessed_data_with_entity_v1.jsonl",
                    "processed_data/entity2sentence_v1.pkl")
