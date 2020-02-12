import json
import re


def read_json(filename, start, end=None):
    ret = []
    with open(filename) as f:
        for i, l in enumerate(f.readlines()):
            if i >= start:
                ret.append(json.loads(l))
            if end is not None and i > end-1:
                break
    return ret


def jsonl_generator(filename):
    with open(filename) as f:
        while True:
            line = f.readline()
            if line != "":
                yield json.loads(line)
            else:
                return ""


def browse_jsonl(filename, step):
    i = 0
    while True:
        yield read_json(filename, start=i, end=i+step)
        i += step


def search_jsonl_file(filename, regex_pattern):
    ret = []
    with open(filename) as f:
        for i, l in enumerate(f.readlines()):
            triple = json.loads(l)
            if re.search(regex_pattern, str(triple)):
                ret.append((i, triple))
    return ret


def list2jsonl_file(dict_list, filename):
    with open(filename, mode="w", encoding="utf-8", errors="ignore") as f:
        for i in dict_list:
            f.write(json.dumps(i) + "\n")


if __name__ == "__main__":
    FileName1 = "supervise_data/supervise_data_v2.jsonl"
    FileName2 = "supervise_data/tagged_corpus_only_true_triple_v1.jsonl"
    FileName3 = "run_LH_code/tagged_corpus.jsonl"
    FileName4 = "processed_data/preprocessed_data_with_entity_v1.jsonl"
    FileName5 = "processed_data/preprocessed_data_with_entity_v2.jsonl"
    FileName = FileName5
    print(f"open {FileName}")
    Browser = browse_jsonl(FileName, 10)
    get1 = search_jsonl_file(FileName, r"\'然而\', \'今晚\', \'是\', \'个\', \'例外\',")
    # f = open("supervise_data/tagged_corpus_with_candidate.jsonl")
    # list_ = json.loads(f.readline())
    # with open("supervise_data/tagged_corpus_with_candidate_v1.jsonl", mode="w") as out_file:
    #     for item in list_:
    #         out_file.write(json.dumps(item) + "\n")




