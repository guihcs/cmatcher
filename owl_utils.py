from rdflib import Graph, URIRef, BNode
from rdflib.namespace import RDF, RDFS, OWL
from om.ont import get_namespace, get_n
from termcolor import colored
import os
from tqdm.auto import tqdm
from os import path
from epc import parser


def parse_tree(tree, g, ng):
    if tree[0] == 'class':
        return parse_class(tree[1], g, ng)
    elif tree[0] == 'restriction' and tree[2] == 'someValuesFrom':
        return parse_restriction(tree[1], tree[3], OWL.someValuesFrom, g, ng)
    elif tree[0] == 'restriction' and tree[2] == 'value':
        return parse_restriction(tree[1], tree[3], OWL.hasValue, g, ng)
    elif tree[0] == 'intersection':
        return parse_intersection(tree[1], g, ng)

    elif tree[0] == 'union':
        return parse_union(tree[1], g, ng)

    elif tree[0] == 'complement':
        return parse_complement(tree[1], g, ng)

    raise NotImplemented('nimp')


def parse_class(tree, g, ng):
    nm = get_namespace(g)
    cl = URIRef(nm + tree)
    ng.add((cl, RDF.type, OWL.Class))
    return cl


def parse_restriction(prop, class_expr, pred, g, ng):
    pp = parse_property(prop, g, ng)
    pc = parse_tree(class_expr, g, ng)

    bn = BNode()
    ng.add((bn, RDF.type, OWL.Restriction))
    ng.add((bn, OWL.onProperty, pp))
    ng.add((bn, pred, pc))
    return bn


def parse_property(tree, g, ng):
    if tree[1] == 'inverse':
        prop = URIRef(get_namespace(g) + tree[2])

        if (prop, OWL.inverseOf, None) in g:
            return g.value(prop, OWL.inverseOf)
        elif (None, OWL.inverseOf, prop) in g:
            raise Exception('have inverse 2')

        else:
            prop = URIRef(get_namespace(g) + tree[2])

            domain = g.value(prop, RDFS.domain)
            rng = g.value(prop, RDFS.range)
            np = URIRef(get_namespace(g) + f'inverseOf_{tree[2]}')
            ng.add((np, RDF.type, OWL.ObjectProperty))
            if rng is not None:
                ng.add((np, RDFS.domain, rng))
            if domain is not None:
                ng.add((np, RDFS.range, domain))
            ng.add((np, OWL.inverseOf, prop))
            return np

    else:
        prop = URIRef(get_namespace(g) + tree[2])

        return prop
    pass


def parse_intersection(tree, g, ng):
    elements = [parse_tree(e, g, ng) for e in tree]
    bn = BNode()
    ng.add((bn, RDF.type, OWL.Class))
    for e in elements:
        ng.add((bn, OWL.intersectionOf, e))
    return bn


def parse_union(tree, g, ng):
    elements = [parse_tree(e, g, ng) for e in tree]
    bn = BNode()
    ng.add((bn, RDF.type, OWL.Class))
    for e in elements:
        ng.add((bn, OWL.unionOf, e))
    return bn


def parse_complement(tree, g, ng):
    e = parse_tree(tree, g, ng)
    bn = BNode()
    ng.add((bn, RDF.type, OWL.Class))
    ng.add((bn, OWL.complementOf, e))
    return bn


def add_depth(r, ng, g, depth, cd=0):
    if cd > depth:
        return
    for s, p, o in g.triples((r, None, None)):
        ng.add((s, p, o))

    for s, p, o in ng.triples((r, None, None)):
        add_depth(o, ng, g, depth, cd + 1)


def pn(n, g, md=2, cd=0):
    print('\t' * (cd * 2), cd, n)
    if cd + 1 > md:
        return

    for s, p, o in g.triples((n, None, None)):
        print('\t' * (cd * 2 + 1), colored(p, 'blue'))
        pn(o, g, md=md, cd=cd + 1)


def to_pyg(tn, ng):
    sm = {tn: 1}
    pm = []
    fm = [[], []]
    for s, p, o in ng:
        if type(s) is BNode:
            sm[s] = 0
        elif s not in sm:
            sm[s] = len(sm) + 1

        if type(o) is BNode:
            sm[o] = 0
        elif o not in sm:
            sm[o] = len(sm) + 1

        pm.append(p)

        fm[1].append(sm[s])
        fm[0].append(sm[o])

    return [x[0] for x in sorted(list(sm.items()), key=lambda x: x[1]) if x[1] != 0], pm, fm




def load_sg(pt, paths):
    data = {}

    graphs = {}

    for p, d, fs in tqdm(list(os.walk(pt))):

        for f in fs:

            if f not in graphs:
                graphs[f] = Graph().parse(paths[f])

            g = graphs[f]
            cqa = p.split('/')[-1]
            if cqa not in data:
                data[cqa] = {}

            with open(path.join(p, f), 'r') as fl:
                text = fl.read()
                tree = parser.parse(text)
                ng = Graph()

                tn = parse_tree(tree, g, ng)

                data[cqa][f] = (tn, ng)

    idata = {}

    for cqa, d in data.items():
        for f, dt in d.items():
            fn = f.split('.')[0]
            if fn not in idata:
                idata[fn] = {}
            idata[fn][cqa] = dt

    return idata




def load_entities(pt, paths):
    data = {}

    graphs = {}

    for p, d, fs in tqdm(list(os.walk(pt))):

        for f in fs:

            if f not in graphs:
                graphs[f] = Graph().parse(paths[f])

            g = graphs[f]
            cqa = p.split('/')[-1]
            if cqa not in data:
                data[cqa] = {}

            with open(path.join(p, f), 'r') as fl:
                text = fl.read()
                tree = parser.parse(text)
                ng = Graph()

                tn = parse_tree(tree, g, ng)

                add_depth(tn, ng, g, 4)

                cm, pm, fm = to_pyg(tn, ng)

                data[cqa][f] = [cm, pm, fm]

    idata = {}

    for cqa, d in data.items():
        for f, dt in d.items():
            fn = f.split('.')[0]
            if fn not in idata:
                idata[fn] = {}
            idata[fn][cqa] = dt

    return idata


def load_cqas(pt):
    data = {}

    for p, d, fs in tqdm(list(os.walk(pt))):

        for f in fs:
            cqa = p.split('/')[-1]
            if cqa not in data:
                data[cqa] = {}

            with open(path.join(p, f), 'r') as fl:
                data[cqa][f] = fl.read()

    idata = {}

    for cqa, d in data.items():
        for f, dt in d.items():
            fn = f.split('.')[0]
            if fn not in idata:
                idata[fn] = {}
            idata[fn][cqa] = dt

    return idata