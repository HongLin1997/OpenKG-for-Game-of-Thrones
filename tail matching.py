import json
import pickle
import re

import pylcs
import Levenshtein
import pyhanlp
from SPARQLWrapper import SPARQLWrapper, JSON
from entity_linking import match_entity
from triple_crawler import sparql_add_triple, sparql_del_triple, sparql_get_name

# Mate Variables
Entity2SentencesFile = "processed_data/entity2sentence_v1.pkl"
CandidateEntityByEditingDistFile = "processed_data/candidate_entity_by_editing_dist_v8.jsonl"
CandidateEntityByEditingDistAndCooccurFile = "processed_data/candidate_entity_by_editing_dist_and_cooccur_v4.jsonl"
EntityReplacementFile = "processed_data/candidate_entity_replacement_list_v4.jsonl"


def jsonl_generator(filename):
    with open(filename) as f:
        while True:
            line = f.readline()
            if line != "":
                yield json.loads(line)
            else:
                return ""


def squeeze_result(res):
    if res["type"] == "uri":
        tmp_split = res["value"].split("/")
        if tmp_split[-2].strip() == "action":
            return f"r:{tmp_split[-1]}"
        elif tmp_split[-2].strip() == "entity":
            return f"e:{tmp_split[-1]}"
    else:
        return res["value"]


def get_all_literal_triples():
    """
    获取所有tail实体为literal的三元组
    :return:
    """
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        f"""
           PREFIX	r:	<http://kg.course/action/>
           PREFIX	e:	<http://kg.course/entity/>
           SELECT DISTINCT *
           WHERE{{
                ?s ?r ?o.
                FILTER regex(str(?r), "不和|兄弟姐妹|分支家族|势力|好友|子嗣|封君|封地|建立者|恋情|执政者|效忠于|母亲|父亲|王室|现任领主|继承人|配偶")
                FILTER isLiteral(?o)
           }}
       """
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    return [{"s": squeeze_result(result["s"]),
             "r": squeeze_result(result["r"]),
             "o": squeeze_result(result["o"])} for result in results]


def get_all_representation():
    """
    获取所有实体对应名称的列表
    :return:
    """
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        f"""
               PREFIX	r:	<http://kg.course/action/>
               PREFIX	e:	<http://kg.course/entity/>
               SELECT DISTINCT ?o
               WHERE{{
                    {{?s r:name ?o.}}
                    UNION{{?s r:别名 ?o}}
               }}
           """
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    return sorted(list(set([squeeze_result(result["o"]) for result in results])))


def triple_word_sim(triple: dict, word: str):
    """
    计算triple和word的相似度
    由triple中的s、o与word的相似度组成
    相似度包含最长子串和莱文斯坦比:
    莱文斯坦比。计算公式 r = (sum – ldist) / sum,
        其中sum是指str1 和 str2 字串的长度总和，
        ldist是类编辑距离。注意这里是类编辑距离，在类编辑距离中删除、插入依然+1，但是替换+2。
    :param triple: dict, triple["s"]为一个实体, triple["r"]为一个关系, triple["o"]为一个literal字符串,
    :param word:
    :return:
    """
    similarity1 = 0  # 与literal的相似度
    similarity2 = 0  # 与头实体的相似度
    s_name = sparql_get_name(triple["s"])
    similarity1 += Levenshtein.ratio(triple["o"], word)
    similarity1 += (pylcs.lcs2(triple["o"], word) * 2) / (len(triple["o"]) + len(word))  # 最长子串
    similarity2 += Levenshtein.ratio(s_name, word)
    similarity2 += (pylcs.lcs2(s_name, word) * 2) / (len(s_name) + len(word))  # 最长子串
    return similarity1 / 2, similarity2 / 2


def k_closest_word_in_editing_dist(triple: dict, word_list: list, k=None, threshold=None):
    """
    找出word_list中与triple相似度前k大个的词语
    :param triple: dict, triple["s"]为一个实体, triple["r"]为一个关系, triple["o"]为一个literal字符串,
    :param word_list: 字符串list
    :param threshold: 距离大于threshold才输出
    :param k:为None时不限制输出词语数
    :return: list of tuple，每个tuple对应word_list中一个词语，以及该词语与word的距离
    """
    ret = []
    for w in word_list:
        ret.append((w, triple_word_sim(triple, w)))
    ret = sorted(ret, key=lambda x: x[1], reverse=True)[:k]
    if threshold:
        ret = [r for r in ret if r[1][0] > threshold or r[1][1] > threshold]
    return ret


def representation2entity(representation):
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        f"""
           PREFIX	r:	<http://kg.course/action/>
           PREFIX	e:	<http://kg.course/entity/>
           SELECT DISTINCT ?s
           WHERE{{
                {{?s r:name "{representation}".}}
                UNION{{?s r:别名 "{representation}"}}
           }}
       """
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    return [squeeze_result(result["s"]) for result in results]


def candidate_entity_by_editing_dist():
    triples = get_all_literal_triples()
    name_list = get_all_representation()
    for i in range(len(triples)):
        print(triples[i])
        closest_word = k_closest_word_in_editing_dist(triples[i], name_list, threshold=0.4)
        if len(closest_word) < 1:
            closest_word = k_closest_word_in_editing_dist(triples[i], name_list, k=1, threshold=0.1)
        closest_entity = [{"entity": e, "editing dist": s}
                          for w, s in closest_word for e in representation2entity(w)
                          if e != triples[i]["s"]]
        triples[i]["candidate_entity"] = closest_entity
    return triples


def list2jsonl_file(dict_list, filename):
    with open(filename, mode="w", encoding="utf-8", errors="ignore") as f:
        for i in dict_list:
            f.write(json.dumps(i) + "\n")


def cal_cooccur(e1_occur, e2_occur):
    """

    :param e1_occur: list of tuple, 每个tuple中为(出现的句子编号, 出现的位置)
    :param e2_occur:
    :return:
    """
    num_cooccur = 0
    occur_sentence1 = set([s[0] for s in e1_occur])
    occur_sentence2 = set([s[0] for s in e2_occur])
    co_occur_sentences = list(occur_sentence1.intersection(occur_sentence2))
    for s_i in co_occur_sentences:
        offsets = set([s[1] for s in e1_occur if s[0] == s_i])
        offsets.update([s[1] for s in e2_occur if s[0] == s_i])
        if len(offsets) > 1:
            num_cooccur += 1
    return num_cooccur


def add_candidate_entity_cooccur():
    ret = []
    i = 0
    with open(Entity2SentencesFile, mode="rb") as f:
        entity2sentence = pickle.load(f)
    for triple in jsonl_generator(CandidateEntityByEditingDistFile):
        e1 = triple["s"]
        for e_i, e in enumerate(triple["candidate_entity"]):
            triple["candidate_entity"][e_i]["cooccur"] = 0
        print(i, triple["s"], triple["r"], triple["o"])
        if e1 not in entity2sentence.keys():
            continue
        i += 1
        for e_i, e in enumerate(triple["candidate_entity"]):
            e2 = e["entity"]
            if e2 not in entity2sentence.keys():
                continue
            cooccur = cal_cooccur(list(entity2sentence[e1]), list(entity2sentence[e2]))
            triple["candidate_entity"][e_i]["cooccur"] = cooccur
        triple["candidate_entity"] = sorted(triple["candidate_entity"], key=lambda x: x["cooccur"], reverse=True)
        print(triple["candidate_entity"])
        ret.append(triple)
    return ret


def cal_domination(a, b, key):
    """
    a如果各个维度上都>=b, 返回True
    :param a:
    :param b:
    :param key:
    :return: a dominate b,  b dominate a
    """
    is_a_dominate = True
    is_b_dominate = True
    for ai, bi in zip(key(a), key(b)):
        if ai < bi:
            is_a_dominate = False
        if bi < ai:
            is_b_dominate = False
    return is_a_dominate, is_b_dominate


def skyline(buildings, key):
    """
    找出skyline点
    :param buildings: 数据点列表
    :param key: 用于比较的维度
    :return:
    """
    skyline_idx = {0}
    for i in range(1, len(buildings)):
        to_drop = set()
        is_dominated = False

        for j in skyline_idx:

            i_dominate, j_dominate = cal_domination(buildings[i], buildings[j], key)

            # Case 1: skyline中的点dominate了新来的点
            if j_dominate:
                is_dominated = True
                break

            # Case 3: 新来的点dominate了skyline中的点, 需要把被dominate的点移除
            if i_dominate:
                to_drop.add(j)

            # Case 2: 其他情况下, 则新来的点是现阶段的skyline点

        if is_dominated:
            continue

        skyline_idx = skyline_idx.difference(to_drop)
        skyline_idx.add(i)
    return skyline_idx


def find_highest_lowerbound(items, key):
    """
    找出同时属于各个属性上的前k个的那一项。不断减小k的值，知道只剩下一个项则返回
    :param key:
    :param items:
    :return:
    """
    num_key = len(key(items[0]))
    sorted_lists = []
    for i in range(num_key):
        l = [i[0] for i in sorted(enumerate(items), key=lambda x: key(x[1])[i], reverse=True)]
        sorted_lists.append(l)
    last_res_set = set()
    for k in range(len(items), 0, -1):
        res_set = set(sorted_lists[0][:k])
        for i in range(1, num_key):
            # 按不同属性值进行排序的列表
            res_set = res_set.intersection(set(sorted_lists[i][:k]))
        if len(res_set) == 1:
            return list(res_set)
        if len(res_set) < 1:
            return list(last_res_set)
        last_res_set = res_set


def replace_list(just_skylines=False):
    """
    每个需要替换literal的三元组，选择候选实体中一个实体记录下来，方便之后进行替换
    :return:
    """
    ret = []
    num_skylines = []
    for triple in jsonl_generator(CandidateEntityByEditingDistAndCooccurFile):
        if len(triple["candidate_entity"]) < 1:
            continue
        if len(triple["candidate_entity"]) == 1:
            triple["candidate_entity"] = triple["candidate_entity"][0]["entity"]
            if just_skylines:
                triple["candidate_entity"] = [triple["candidate_entity"]]
            ret.append(triple)
            continue
        # 如果有实体与尾实体完全匹配, 则直接选择该实体
        most_matched_with_tail = max(triple["candidate_entity"], key=lambda x: x["editing dist"][0])
        if most_matched_with_tail["editing dist"][0] == 1:
            triple["candidate_entity"] = most_matched_with_tail["entity"]
            if just_skylines:
                triple["candidate_entity"] = [triple["candidate_entity"]]
            ret.append(triple)
            continue
        skyline_entities_idx = skyline(triple["candidate_entity"],
                                       key=lambda x: (x["editing dist"][0], x["editing dist"][1], x["cooccur"]))
        skyline_entities_idx = list(skyline_entities_idx)
        skyline_entities = [triple["candidate_entity"][i] for i in skyline_entities_idx]
        num_skylines.append(len(skyline_entities_idx))
        if just_skylines:
            triple["candidate_entity"] = [e["entity"] for e in skyline_entities]
            ret.append(triple)
            continue
        if len(skyline_entities_idx) == 1:
            triple["candidate_entity"] = skyline_entities[0]["entity"]
            ret.append(triple)
            continue
        else:
            lowerbound = find_highest_lowerbound(skyline_entities,
                                                 key=lambda x:
                                                 (x["editing dist"][0], x["editing dist"][1], x["cooccur"]))
            if len(lowerbound) == 1:
                # 通过lowerbound能找出唯一一个下界
                triple["candidate_entity"] = skyline_entities[lowerbound[0]]["entity"]
            else:
                # 通过lowerbound仍无法得到唯一一个结果, 则只使用与尾实体的相似度已经与头实体的共现来找下界
                next_lowerbound = find_highest_lowerbound(skyline_entities,
                                                          key=lambda x: (x["editing dist"][0], x["cooccur"]))
                if lowerbound == 1:
                    triple["candidate_entity"] = skyline_entities[next_lowerbound[0]]["entity"]
                else:
                    # 仍是没有唯一结果，则使用与尾实体相似度最高的那个
                    triple["candidate_entity"] = \
                        max(skyline_entities,
                            key=lambda x: x["editing dist"][0])["entity"]
                ret.append(triple)
    return ret, num_skylines


def do_replacement():
    for triple in jsonl_generator("processed_data/candidate_entity_replacement_list_v6.jsonl"):
        sparql_del_triple(triple["s"], triple["r"], "\"" + triple["o"] + "\"")
        sparql_add_triple(triple["s"], triple["r"], triple["candidate_entity"])


def search_jsonl_file(filename, regex_pattern):
    ret = []
    with open(filename) as f:
        for l in f.readlines():
            triple = json.loads(l)
            if re.search(regex_pattern, str(triple)):
                ret.append(triple)
    return ret


if __name__ == "__main__":
    # 1. 从图谱中找出tail需要匹配为实体的那些三元组
    # CandidateEntityByEditingDist = candidate_entity_by_editing_dist()
    # list2jsonl_file(CandidateEntityByEditingDist, "processed_data/candidate_entity_by_editing_dist_v8.jsonl")
    # # 统计没有匹配实体的literal, 以及各个匹配到的literal
    # num_empty_candidate = 0
    # candidate_len = []
    # for i in CandidateEntityByEditingDist:
    #     candidate_len.append(len(i["candidate_entity"]))
    #     if len(i["candidate_entity"]) == 0:
    #         num_empty_candidate += 1
    # most_entity_literal = CandidateEntityByEditingDist[candidate_len.index(max(candidate_len))]["o"]
    # print(f"共有{len(CandidateEntityByEditingDist)}个literal,"
    #       f"有{num_empty_candidate}个literal没有匹配到实体, "
    #       f"每个literal平均匹配到{sum(candidate_len)/len(candidate_len)},"
    #       f"一个literal({most_entity_literal})最多匹配到{max(candidate_len)}个实体")
    # 2. 找出实体的共现次数
    # CandidateEntityByEditingDistAndCooccur = add_candidate_entity_cooccur()
    # list2jsonl_file(CandidateEntityByEditingDistAndCooccur,
    #                 "processed_data/candidate_entity_by_editing_dist_and_cooccur_v4.jsonl")
    # # 搜索jsonl文件
    triples = search_jsonl_file("processed_data/candidate_entity_replacement_list_v5.jsonl", "托曼一世")
    # 3. 选取出可替换的实体
    # Replacement, NSkyline = replace_list(just_skylines=False)
    # list2jsonl_file(Replacement, "processed_data/candidate_entity_replacement_list_v6.jsonl")
    # do_replacement()
