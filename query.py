from SPARQLWrapper import SPARQLWrapper, JSON, N3

if __name__ == "__main__":
    sparql = SPARQLWrapper("http://localhost:3030/testds/sparql")
    sparql.setQuery(
        """
        PREFIX	r:	<http://kg.course/action/>
        PREFIX	e:	<http://kg.course/entity/>
        SELECT DISTINCT ?s ?r ?o
        WHERE {
            ?s ?r ?o.
        }
        """
    )
    sparql.setReturnFormat(N3)
    # sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        print(result["o"]["value"])
