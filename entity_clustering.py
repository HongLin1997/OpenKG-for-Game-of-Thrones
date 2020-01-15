import sparql_utils

def index_entities():
    id2entity = sparql_utils.list_all_entity()
    entity2id = dict()
    for i, e in enumerate(id2entity):
        entity2id[e] = i
    return id2entity, entity2id


if __name__ == "__main__":
    # 获取entity2id和id2entity
    ID2E, E2ID = index_entities()
    # 构造邻接矩阵