import json
import sqlite3
import pickle

# Mate Variables
from SPARQLWrapper import SPARQLWrapper, JSON

SentenceFile = "processed_data/preprocessed_data_with_entity_v1.jsonl"
Entity2SentencesFile = "processed_data/entity2sentence_v1.pkl"
EntityReplacementFile = "processed_data/candidate_entity_replacement_list_v5.jsonl"
SQLiteFile = "processed_data/weak_supervise.db"
SuperviseDataFile = "processed_data/supervise_data_v2.jsonl"

def squeeze_result(res):
    if res["type"] == "uri":
        tmp_split = res["value"].split("/")
        if tmp_split[-2].strip() == "action":
            return f"r:{tmp_split[-1]}"
        elif tmp_split[-2].strip() == "entity":
            return f"e:{tmp_split[-1]}"
    else:
        return res["value"]


def create_table():
    conn = sqlite3.connect(SQLiteFile)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE data_index
           (sentenceID INT NOT NULL,
            entity1    TEXT    NOT NULL,
            entity2    TEXT    NOT NULL,
            relation    TEXT    NOT NULL,
            offset1    INT     NOT NULL,
            offset2    INT     NOT NULL,
           );
        """
    )
    conn.commit()
    conn.close()


def insert_data(in_tuple):
    conn = sqlite3.connect(SQLiteFile)
    c = conn.cursor()
    c.execute(
        "INSERT INTO data_index (sentenceID, entity1, entity2, relation, offset1, offset2) "
        "VALUES (?, ?, ?, ?, ?, ?)", in_tuple
    )
    conn.commit()
    conn.close()


def sparql_get_all_two_entities_triple():
    """
    返回所有头尾实体都不是literal的triple
    :return:
    """
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        """
            PREFIX	r:	<http://kg.course/action/>
            PREFIX	e:	<http://kg.course/entity/>
            SELECT DISTINCT *
            WHERE {
                ?s ?r ?o.
                MINUS{
                    ?s ?r ?o
                    FILTER isLiteral(?o)
                }
            }
        """
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [(squeeze_result(r["s"]), squeeze_result(r["r"]), squeeze_result(r["o"]))
            for r in results["results"]["bindings"]]


def list2jsonl_file(dict_list, filename):
    with open(filename, mode="w", encoding="utf-8", errors="ignore") as f:
        for i in dict_list:
            f.write(json.dumps(i) + "\n")


def jsonl_generator(filename):
    with open(filename) as f:
        while True:
            line = f.readline()
            if line != "":
                yield json.loads(line)
            else:
                return ""


def combine_offset(list1, list2):
    """
    将list1和list2的元素两两组合
    :param list1:
    :param list2:
    :return:
    """
    ret = []
    for i in list1:
        for j in list2:
            if i != j:
                ret.append((i, j))
    return ret


def find_sentence(e1_occur, e2_occur):
    """

    :param e1_occur: list of tuple, 每个tuple中为(出现的句子编号, 出现的位置)
    :param e2_occur:
    :return:
    """
    ret = []
    occur_sentence1 = set([s[0] for s in e1_occur])
    occur_sentence2 = set([s[0] for s in e2_occur])
    co_occur_sentences = list(occur_sentence1.intersection(occur_sentence2))
    for s_i in co_occur_sentences:
        offsets1 = [s[1] for s in e1_occur if s[0] == s_i and s[1] != "pov"]
        offsets2 = [s[1] for s in e2_occur if s[0] == s_i and s[1] != "pov"]
        ret.append({"sentenceID": s_i, "offsets": combine_offset(offsets1, offsets2)})
    return ret


def construct_weak_supervise_data():
    ret = []  # tuple列表, 每个tuple=(sentenceID, entity1, offset1, entity2, offset2, relation)
    with open(Entity2SentencesFile, mode="rb") as f:
        entity2sentence = pickle.load(f)
    # replacement list里的三元组
    for triple in jsonl_generator(EntityReplacementFile):
        s_entity = triple["s"]
        r = triple["r"]
        o_literal = triple["o"]
        print(f"{s_entity}, {r}, {o_literal}")
        o_entities = triple["candidate_entity"]
        for o in o_entities:
            if s_entity not in entity2sentence.keys() or o not in entity2sentence.keys():
                continue
            for cooccur_info in find_sentence(entity2sentence[s_entity], entity2sentence[o]):
                for offset1, offset2 in cooccur_info["offsets"]:
                    ret.append((cooccur_info["sentenceID"],
                                s_entity, offset1,
                                o_literal, offset2,
                                r))
    for triple in sparql_get_all_two_entities_triple():
        s_entity = triple[0]
        r = triple[1]
        o_entity = triple[2]
        print(f"{s_entity}, {r}, {o_entity}")
        if s_entity not in entity2sentence.keys() or o_entity not in entity2sentence.keys():
            continue
        for cooccur_info in find_sentence(entity2sentence[s_entity], entity2sentence[o_entity]):
            for offset1, offset2 in cooccur_info["offsets"]:
                ret.append((cooccur_info["sentenceID"],
                            s_entity, offset1,
                            o_entity, offset2,
                            r))
    ret = list(set(ret))
    return ret


def instantiate_sentence_in_superivse_data(supervise_data):
    data_ptr = 0
    ret = []
    for i, s in enumerate(jsonl_generator(SentenceFile)):
        data = supervise_data[data_ptr]
        while data[0] == i:
            print(i, data_ptr)
            ret.append({
                "sentence": s["hanlp_tokens"],
                "s": data[1], "s_offset": data[2],
                "o": data[3], "o_offset": data[4],
                "r": data[5]
            })
            data_ptr += 1
            if data_ptr >= len(supervise_data):
                break
            data = supervise_data[data_ptr]
        if data_ptr >= len(supervise_data):
            break

    return ret


if __name__ == "__main__":
    SuperviseData = construct_weak_supervise_data()
    SuperviseData = sorted(SuperviseData, key=lambda x: x[0])
    SuperviseData = instantiate_sentence_in_superivse_data(SuperviseData)

    list2jsonl_file(SuperviseData, SuperviseDataFile)
