from rdflib import Graph
from rdflib.term import URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
from om.ont import get_n, tokenize


def get_parents(e, g: Graph, max_iterations=50, max_entities=50):
    parents = []
    for _ in range(max_iterations):
        p = g.value(e, RDFS.subClassOf)
        if len(parents) >= max_entities:
            break
        if p is None:
            break
        parents.append(p)
        e = p

    return parents


def get_children(e, g: Graph, max_entities=50):
    children = []

    for s, p, o in g.triples((None, RDFS.subClassOf, e)):
        children.append(s)
        if len(children) >= max_entities:
            break

    return children


def get_incoming_properties(e, g: Graph, max_entities=50):
    properties = []

    for s, p, o in g.triples((None, None, e)):
        properties.append((s, p))
        if len(properties) >= max_entities:
            break

    return properties


def get_outgoing_properties(e, g: Graph, max_entities=50):
    properties = []

    for s, p, o in g.triples((e, None, None)):
        if p in {OWL.disjointWith, RDFS.subClassOf, OWL.maxCardinality, OWL.onProperty}:
            continue
        if type(o) == Literal and o.language is not None and o.language != 'en':
            continue
        properties.append((p, o))
        if len(properties) >= max_entities:
            break

    return properties


def get_disjoints(e, g: Graph, max_entities=50):
    disjoints = []

    for s, p, o in g.triples((e, OWL.disjointWith, None)):
        disjoints.append(o)
        if len(disjoints) >= max_entities:
            break

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

    parents = list_to_text_form(get_parents(e, g, max_entities=max_entities), g)
    children = list_to_text_form(get_children(e, g, max_entities=max_entities), g)
    incoming_properties = list_to_text_form(get_incoming_properties(e, g, max_entities=max_entities), g)
    outgoing_properties = list_to_text_form(get_outgoing_properties(e, g, max_entities=max_entities), g)
    disjoints = list_to_text_form(get_disjoints(e, g, max_entities=max_entities), g)

    return f'Target Concept:\nName: {entity_text}.\nParents: {parents}.\nChildren: {children}.\nIncoming Properties: {incoming_properties}.\nOutgoing Properties: {outgoing_properties}.\nDisjoint with: {disjoints}.'
