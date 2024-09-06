from rdflib import Graph
from rdflib.term import URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS

def page_rank(g, ents, num_iterations=10, damping_factor=0.85):
    pr = {e: 1/ len(ents) for e in ents}

    for _ in range(num_iterations):
        new_pr = {e: 0 for e in ents}
        for e in ents:
            incoming_links = find_incoming_links(e, g)
            sum_rank = 0
            for l in incoming_links:
                lo = count_outgoing_links(l, g)
                sum_rank += damping_factor * pr[l] / lo + (1 - damping_factor) / len(ents)

            new_pr[e] = sum_rank

        pr = new_pr

    return pr


def find_incoming_links(e, g):
    incoming_links = set()

    for s, p, o in g.triples((None, None, e)):
        incoming_links.add(s)
    return incoming_links


def count_outgoing_links(e, g):
    objects = set()
    for s, p, o in g.triples((e, None, None)):
        objects.add(o)
    return len(objects)


def gen_pagerank_sparql_queries(g, num_iterations=10, damping_factor=0.8, threshold=0.4, max_entities=30):

    ents = set(g.subjects())
    ranks = page_rank(g, ents, num_iterations=num_iterations, damping_factor=damping_factor)

    values = list(sorted(ranks.items(), key=lambda x: x[1], reverse=True))

    _, bv = values[0]

    fv = list(filter(lambda x: x[1] / bv > threshold, values))

    queries = []

    for k, v in fv[:max_entities]:
        kv = g.value(k, RDF.type)
        if kv is None:
            continue
        if kv == OWL.Class:
            queries.append(f'SELECT DISTINCT ?x WHERE {{?x a <{k}>.}}')
        elif 'property' in kv.lower():
            queries.append(f'SELECT DISTINCT ?x ?y WHERE {{?x <{k}> ?y.}}')
        else:
            queries.append(f'SELECT DISTINCT ?x WHERE {{?x a <{k}>.}}')

    return queries
