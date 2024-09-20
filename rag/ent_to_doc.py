from rdflib import Graph
from rdflib.term import URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from om.ont import get_n, tokenize


def get_outgoing_links(e, g):
    links = []
    for s, p, o in g.triples((e, None, None)):
        links.append((p, o))

    for s, p, o in g.triples((None, None, e)):
        if p not in {RDFS.domain}:
            continue
        links.append((s, g.value(s, RDFS.range)))

    return links


def get_incoming_links(e, g):
    links = []

    for s, p, o in g.triples((None, None, e)):
        if p in {RDFS.domain}:
            continue
        if p in {RDFS.range}:
            links.append((g.value(s, RDFS.domain), s))
        else:
            links.append((s, p))

    return links


def filter_properties(properties, i, values):
    return [p for p in properties if p[i] not in values]


def get_out_property_relevance(p, o1):
    l_out = filter_properties(get_outgoing_links(p[1], o1), 0, {OWL.disjointWith, RDF.type})
    l_in = filter_properties(get_incoming_links(p[1], o1), 1, {OWL.disjointWith, RDF.type})
    rdiv = len(l_out) + len(l_in)
    return 1 / rdiv if rdiv > 0 else 1


def get_in_property_relevance(p, o1):
    l_out = filter_properties(get_outgoing_links(p[0], o1), 1, {OWL.disjointWith, RDF.type})
    l_in = filter_properties(get_incoming_links(p[0], o1), 0, {OWL.disjointWith, RDF.type})
    rdiv = len(l_out) + len(l_in)
    return len(l_in) / rdiv if rdiv > 0 else 1


def get_connection_relevance(p, o1):
    l_out = filter_properties(get_outgoing_links(p, o1), 1, {OWL.disjointWith, RDF.type})
    l_in = filter_properties(get_incoming_links(p, o1), 0, {OWL.disjointWith, RDF.type})

    rdiv = len(l_out) + len(l_in)
    return rdiv


def get_parents(e, g: Graph, max_iterations=50):
    parents = []
    for _ in range(max_iterations):
        p = g.value(e, RDFS.subClassOf)

        if p is None:
            break
        parents.append(p)
        e = p

    return parents


def get_children(e, g: Graph):
    children = []

    for s, p, o in g.triples((None, RDFS.subClassOf, e)):
        children.append(s)

    return children


def get_incoming_properties(e, g: Graph):
    properties = []

    for s, p, o in g.triples((None, None, e)):
        properties.append((s, p))

    return properties


def get_outgoing_properties(e, g: Graph):
    properties = []

    for s, p, o in g.triples((e, None, None)):
        if p in {OWL.disjointWith, RDFS.subClassOf, OWL.maxCardinality, OWL.onProperty}:
            continue
        if type(o) == Literal and o.language is not None and o.language != 'en':
            continue
        properties.append((p, o))

    return properties


def get_disjoints(e, g: Graph):
    disjoints = []

    for s, p, o in g.triples((e, OWL.disjointWith, None)):
        disjoints.append(o)

    return disjoints


def to_text_form(e, g, max_len=125):
    if type(e) == BNode:
        return 'BNode'

    if type(e) == Literal:
        return str(e)[:max_len]
    return ' '.join(tokenize(get_n(e, g)))[0:max_len]


def list_to_text_form(l, g):
    if len(l) == 0:
        return 'Empty'

    if type(l[0]) == tuple:
        return ', '.join([f'{to_text_form(e[0], g)} {to_text_form(e[1], g)}' for e in l])

    return ', '.join([to_text_form(e, g) for e in l])


def flat_list(e, g: Graph, max_it=50):
    res = []

    for _ in range(max_it):

        if e is None:
            break
        f = g.value(e, RDF.first)
        if f is None:
            break
        res.append(f)

        e = g.value(e, RDF.rest)

    return res


def gen_doc(e, g: Graph, max_entities=50, max_it=50):
    if type(e) != BNode:
        entity_text = to_text_form(e, g)
    else:

        if g.value(e, RDF.type) == OWL.Restriction:

            props = []

            for s, p, o in g.triples((e, None, None)):
                if p == RDF.type:
                    continue
                props.append((p, o))

            props_text = ', '.join([f'{to_text_form(p, g)} = {to_text_form(o, g)}' for p, o in props])
            entity_text = f'Restriction({props_text})'


        elif (e, OWL.unionOf, None) in g:
            v = g.value(e, OWL.unionOf)
            fl = flat_list(v, g)
            entity_text = f'UnionOf({list_to_text_form(fl, g)})'

        elif (e, OWL.intersectionOf, None) in g:
            v = g.value(e, OWL.intersectionOf)
            fl = flat_list(v, g)
            entity_text = f'IntersectionOf({list_to_text_form(fl, g)})'

        elif (e, OWL.oneOf, None) in g:
            v = g.value(e, OWL.oneOf)
            fl = flat_list(v, g)
            entity_text = f'OneOf({list_to_text_form(fl, g)})'

        elif (e, OWL.complementOf, None) in g:
            v = g.value(e, OWL.complementOf)
            entity_text = f'ComplementOf({to_text_form(v, g)})'

        elif (e, OWL.distinctMembers, None) in g:

            v = g.value(e, OWL.distinctMembers)
            fl = flat_list(v, g)
            entity_text = f'DistinctMembers({list_to_text_form(fl, g)})'

        elif (e, OWL.members, None) in g:

            v = g.value(e, OWL.members)
            fl = flat_list(v, g)
            entity_text = f'Members({list_to_text_form(fl, g)})'

        elif (e, OWL.inverseOf, None) in g:

            v = g.value(e, OWL.inverseOf)
            entity_text = f'InverseOf({to_text_form(v, g)})'

        elif (e, RDF.type, OWL.Axiom) in g:
            entity_text = 'Axiom'

        elif (e, RDF.type, URIRef('http://www.w3.org/2003/11/swrl#Imp')) in g:

            entity_text = 'Imp'

        elif (e, URIRef('http://www.w3.org/2003/11/swrl#classPredicate'), None) in g:

            entity_text = 'ClassPredicate'

        elif (e, URIRef('http://www.w3.org/2003/11/swrl#propertyPredicate'), None) in g:

            entity_text = 'PropertyPredicate'
        else:
            entity_text = 'BNode'

    parents = get_parents(e, g)
    children = get_children(e, g)

    if len(children) > max_entities:
        ranked_children = [(p, get_connection_relevance(p, g)) for p in children]
        ranked_children.sort(key=lambda x: x[1], reverse=True)
        children = [p[0] for p in ranked_children[:max_entities]]

    incoming_properties = filter_properties(get_incoming_links(e, g), 1, {OWL.disjointWith, RDF.type, RDFS.subClassOf})

    if len(incoming_properties) > max_entities:
        ranked_incoming_properties = [(p, get_in_property_relevance(p, g)) for p in incoming_properties]
        ranked_incoming_properties.sort(key=lambda x: x[1], reverse=True)
        incoming_properties = [p[0] for p in ranked_incoming_properties[:max_entities]]

    outgoing_properties = filter_properties(get_outgoing_links(e, g), 0, {OWL.disjointWith, RDF.type, RDFS.subClassOf})
    disjoints = get_disjoints(e, g)

    if len(disjoints) > max_entities:
        ranked_disjoints = [(p, get_connection_relevance(p, g)) for p in disjoints]
        ranked_disjoints.sort(key=lambda x: x[1], reverse=True)
        disjoints = [p[0] for p in ranked_disjoints[:max_entities]]

    # filter nonetype values

    parents = [p for p in parents if p is not None]
    children = [p for p in children if p is not None]
    incoming_properties = [p for p in incoming_properties if p[0] is not None]
    outgoing_properties = [p for p in outgoing_properties if p[1] is not None]
    disjoints = [p for p in disjoints if p is not None]

    parents.sort(key=lambda x: get_n(x, g))
    children.sort(key=lambda x: get_n(x, g))
    incoming_properties.sort(key=lambda x: get_n(x[0], g))
    outgoing_properties.sort(key=lambda x: get_n(x[1], g))
    disjoints.sort(key=lambda x: get_n(x, g))

    parents = list_to_text_form(parents, g)
    children = list_to_text_form(children, g)
    incoming_properties = list_to_text_form(incoming_properties, g)
    outgoing_properties = list_to_text_form(outgoing_properties, g)
    disjoints = list_to_text_form(disjoints, g)

    return f'Target Concept:\nName: {entity_text}.\nParents: {parents}.\nChildren: {children}.\nIncoming Properties: {incoming_properties}.\nOutgoing Properties: {outgoing_properties}.\nDisjoint with: {disjoints}.'
