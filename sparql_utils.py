import os

from SPARQLWrapper import SPARQLWrapper, JSON


INCRE_PATH = "triples/incremental files"


def squeeze_result(res):
    if res["type"] == "uri":
        tmp_split = res["value"].split("/")
        if tmp_split[-2].strip() == "action":
            return f"r:{tmp_split[-1]}"
        elif tmp_split[-2].strip() == "entity":
            return f"e:{tmp_split[-1]}"
    else:
        return res["value"]


def add_triple(s, p, o):
    sparql = SPARQLWrapper("http://localhost:3030/testds/update")
    try:
        sparql.setQuery(
            f"""
                PREFIX	r:	<http://kg.course/action/>
                PREFIX	e:	<http://kg.course/entity/>
                INSERT DATA {{{s} {p} {o}.}}
            """
        )
        sparql.method = "POST"
        sparql.query()
    except BaseException as e:
        with open(os.path.join(INCRE_PATH, "sparqlerror.log"), mode="a", encoding="utf-8", errors="ignore") as f:
            f.write(f"{e}\n")


def del_triple(s, p, o):
    sparql = SPARQLWrapper("http://localhost:3030/testds/update")
    try:
        sparql.setQuery(
            f"""
                PREFIX	r:	<http://kg.course/action/>
                PREFIX	e:	<http://kg.course/entity/>
                DELETE WHERE{{{s} {p} {o}.}}
            """
        )
        sparql.method = "POST"
        sparql.query()
    except BaseException as e:
        with open(os.path.join(INCRE_PATH, "sparqlerror.log"), mode="a", encoding="utf-8", errors="ignore") as f:
            f.write(f"{e}\n")


def list_all_nonliteral():
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        f"""
            PREFIX	r:	<http://kg.course/action/>
            PREFIX	e:	<http://kg.course/entity/>
            SELECT DISTINCT ?e
            WHERE {{
                {{?e ?r ?o}}
                UNION{{
                    ?o ?r ?e
                    MINUS{{
                        ?o ?r ?e
                        FILTER isLiteral(?e)
                    }}
                }}
            }}
        """
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [squeeze_result(r["e"]) for r in results["results"]["bindings"]]


def list_noliteral_triples():
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        f"""
            PREFIX	r:	<http://kg.course/action/>
            PREFIX	e:	<http://kg.course/entity/>
            SELECT DISTINCT *
            WHERE {{
                {{?s ?r ?o}}   
                MINUS{{
                    ?s ?r ?o
                    FILTER isLiteral(?o)
                }}
            }}
        """
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [(squeeze_result(r["s"]), squeeze_result(r["r"]), squeeze_result(r["o"]))
            for r in results["results"]["bindings"]]


def get_all_entity(category=None):
    """
    获取所有实体
    :return:
    """
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    if category is not None:
        sparql.setQuery(
            f"""
                   PREFIX	r:	<http://kg.course/action/>
                   PREFIX	e:	<http://kg.course/entity/>
                   SELECT DISTINCT ?s
                   WHERE{{
                        {{?s r:type "{category}".}}
                   }}
               """
        )
    else:
        sparql.setQuery(
            f"""
                           PREFIX	r:	<http://kg.course/action/>
                           PREFIX	e:	<http://kg.course/entity/>
                           SELECT DISTINCT ?s
                           WHERE{{
                                {{?s r:type ?o.}}
                           }}
                       """
        )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()["results"]["bindings"]
    return sorted(list(set([squeeze_result(result["s"]) for result in results])))


def relation_between_two_entity(e1, e2):
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        f"""
                PREFIX	r:	<http://kg.course/action/>
                PREFIX	e:	<http://kg.course/entity/>
                SELECT DISTINCT ?r
                WHERE {{
                    {e1} ?r {e2}.
                }}
            """
    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return [squeeze_result(r["r"]) for r in results["results"]["bindings"]]
