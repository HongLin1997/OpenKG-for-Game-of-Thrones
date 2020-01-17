import collections
import json
import random

import matplotlib
import numpy
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt

import sparql_utils


def index_entities(category=None) -> (list, dict):
    """
    获取实体的索引
    :return:
    """
    if category is None:
        id2entity = sparql_utils.list_all_nonliteral()
    else:
        id2entity = sparql_utils.get_all_entity(category)
    entity2id = dict()
    for i, e in enumerate(id2entity):
        entity2id[e] = i
    return id2entity, entity2id


def count_relationship(entity2id) -> dict:
    """
    计算各个实体之间的关系数量
    :return:
        relation_count: dict, key=tuple(e1, e2), value=num of relations
        num_entity: 实体的数量
    """
    triples = sparql_utils.list_noliteral_triples()
    relation_count = dict()
    print("计数所有头尾都是实体的三元组")
    for s, r, o in tqdm(triples):
        if s in entity2id.keys() and o in entity2id.keys():
            relation_count[(entity2id[s], entity2id[o])] = relation_count.get((entity2id[s], entity2id[o]), 0) + 1
            relation_count[(entity2id[o], entity2id[s])] = relation_count.get((entity2id[o], entity2id[s]), 0) + 1
    return relation_count


def construct_affinity_mat(entity2id: dict, num_entity: int) -> coo_matrix:
    """
    获取实体之间的邻接矩阵
    :return:
    """
    relation_count = count_relationship(entity2id)
    row_idx = []
    col_idx = []
    affinity = []
    print("计数结果存入矩阵")
    for k, v in tqdm(relation_count.items()):
        row_idx.append(k[0])
        col_idx.append(k[1])
        affinity.append(v)
    return coo_matrix((affinity, (row_idx, col_idx)), shape=(num_entity, num_entity))


def page_rank(m: csr_matrix, alpha):
    """
    """
    n, _ = m.get_shape()
    degree = numpy.sum(m, axis=1)
    weight_mat = m / (degree + numpy.array(degree == 0, dtype=int))
    # weight_mat = m
    v = numpy.random.rand(n).reshape((-1, 1))
    last_v = v
    while True:
        v = alpha * (weight_mat.T * v) + (1 - alpha) * v
        delta = numpy.sum(abs(v-last_v))
        print(delta)
        if delta < 0.0001:
            break
        last_v = v
    return numpy.array(v).flatten()


def draw_graph(graph: coo_matrix, id2entity: list, node_labels=None, node_rank=None, output_path=None):
    candidate_colors = ["lightblue", "orange", "Green", "tomato", "gold", "black", "hotpink",  "chocolate"]
    label_cntr = collections.Counter()
    for i, lb in enumerate(node_labels):
        label_cntr[lb] += node_rank[i]
    label_cntr = sorted(label_cntr.items(), key=lambda x: x[1], reverse=True)
    colors = [""] * len(label_cntr)
    for i, (lb, cnt) in enumerate(label_cntr):
        colors[lb] = candidate_colors[i]
    node_colors = None
    if node_labels is not None:
        node_colors = [colors[lb] for lb in node_labels]
    nx_graph = nx.Graph()
    for i in range(len(id2entity)):
        nx_graph.add_node(i)
    nx_graph.add_weighted_edges_from([(int(node1), int(node2), int(edge_weight))
                                      for node1, node2, edge_weight in zip(graph.row, graph.col, graph.data)])
    if output_path is not None:
        json_data = json_graph.node_link_data(nx_graph)
        for i in range(len(id2entity)):
            json_data["nodes"][i]["name"] = id2entity[json_data["nodes"][i]["id"]]
            json_data["nodes"][i]["color"] = node_colors[i]
            json_data["nodes"][i]["radius"] = node_rank[i] + 0.05
        with open(output_path, mode="w", encoding="utf-8", errors="ignore") as f:
            json.dump(json_data, f)
    else:
        nx_graph = nx.relabel_nodes(nx_graph, {i: e for i, e in enumerate(id2entity)})
        nx.draw(nx_graph, with_labels=True, font_weight='bold', node_color=node_colors)
        plt.show()


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'sans-serif']
    # 获取entity2id和id2entity
    Category = None
    ID2E, E2ID = index_entities(Category)
    NumEntity = len(ID2E)
    # 构造邻接矩阵
    GOT_KG = construct_affinity_mat(E2ID, NumEntity)
    # 聚类
    Clustering = SpectralClustering(n_clusters=8, affinity='precomputed',
                                    assign_labels="discretize", random_state=0, n_jobs=4).fit(GOT_KG)
    # page rank
    EntityRank = page_rank(GOT_KG.tocsr(), 0.99)
    ImportantEntityIdx = numpy.argsort(EntityRank)[::-1]
    ImportantEntity = [ID2E[idx] for idx in ImportantEntityIdx]
    # shuffle index
    ShuffleIndex = list(range(len(ID2E)))
    random.shuffle(ShuffleIndex)
    ShuffleIndex = ShuffleIndex[:50]
    # get a minified graph
    MinIndex = ImportantEntityIdx[:]
    GOT_KG_min = GOT_KG.tocsr()[MinIndex, :][:, MinIndex].tocoo()
    ID2E_min = [ID2E[idx] for idx in MinIndex]
    Labels_min = [Clustering.labels_[idx] for idx in MinIndex]
    # output graph
    draw_graph(GOT_KG, ID2E, Clustering.labels_, EntityRank, output_path=f"graphs_json/GOT_graph_{Category}.json")  # 输出到json
    # draw_graph(GOT_KG_min, ID2E_min, Labels_min)  # 直接显示
