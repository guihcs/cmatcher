import itertools
import torch
import random
from torch_geometric.data import Data
from rdflib.term import BNode


def build_raw_data(idata, cqas):
    '''

    :param idata:
    :param cqas:
    :return: (cqa name, anchor cqa, positive cqa, anchor entity, positive entity)
    '''

    raw_data = {}

    for ts in idata.keys():
        ks = set(idata.keys())
        ks.remove(ts)
        ds = []
        for o1, o2 in itertools.permutations(ks, 2):

            for cqa in idata[o1]:
                if cqa not in idata[o2]:
                    continue

                ds.append((cqa, cqas[o1][cqa], cqas[o2][cqa], idata[o1][cqa], idata[o2][cqa]))

        raw_data[ts] = ds

    return raw_data


def get_dataset_stats(raw_data):
    max_len_cqa = 0
    max_feature_count = 0
    max_feature_len = 0
    max_property_count = 0
    max_property_len = 0
    max_edge_count = 0
    for cqa, c1, c2, e1, e2 in raw_data:
        if len(c1) > max_len_cqa:
            max_len_cqa = len(c1)

        if len(e1[0]) > max_feature_count:
            max_feature_count = len(e1[0])

        m1 = max(map(len, e1[0]))
        if m1 > max_feature_len:
            max_feature_len = m1

        if len(e1[1]) > max_property_count:
            max_property_count = len(e1[1])

        m2 = max(map(len, e1[1]))
        if m2 > max_property_len:
            max_property_len = m2

        if len(e1[2]) > max_edge_count:
            max_edge_count = len(e1[2])

    return {'max_len_cqa': max_len_cqa, 'max_feature_count': max_feature_count, 'max_feature_len': max_feature_len,
            'max_property_count': max_property_count, 'max_property_len': max_property_len,
            'max_edge_count': max_edge_count}


def pad_seq(t, max_len):
    return torch.cat([t, torch.zeros((t.shape[0], max_len - t.shape[1]))], dim=1)


def pad_seq2(t, max_len):
    return torch.cat([t, torch.zeros((max_len - t.shape[0], t.shape[1]))], dim=0)


def pad_edge(t, max_len):
    return t + (max_len - len(t)) * [[-1, -1]]


def pad_entities(tokenizer, entities, ml):
    ft = list(map(lambda x: x if type(x) is not BNode else 'blank node', entities))
    sm = {}
    n = []
    for i, f in enumerate(ft):
        if f not in sm:
            sm[f] = len(sm)
            n.append(f)

        ft[i] = sm[f]
    e1id = tokenizer(n, return_tensors='pt', padding=True)['input_ids']
    return pad_seq(e1id, ml), torch.LongTensor(ft)


class PadData:

    def __init__(self, cqa_name, anchor_cqa, positive_cqa, anchor_entities, anchor_entities_index,
                 positive_entities,
                 positive_entities_index, anchor_properties, anchor_properties_index, positive_properties,
                 positive_properties_index, edge1, edge2):
        self.cqa_name = cqa_name
        self.anchor_cqa = anchor_cqa
        self.positive_cqa = positive_cqa
        self.anchor_entities = anchor_entities
        self.anchor_entities_index = anchor_entities_index
        self.positive_entities = positive_entities
        self.positive_entities_index = positive_entities_index
        self.anchor_properties = anchor_properties
        self.anchor_properties_index = anchor_properties_index
        self.positive_properties = positive_properties
        self.positive_properties_index = positive_properties_index
        self.edge1 = edge1
        self.edge2 = edge2


def pad_dataset(tokenizer, raw_data, stats) -> list[PadData]:
    pd = []
    for cqa_name, anchor_cqa, positive_cqa, anchor_graph, positive_graph in raw_data:
        ids1 = tokenizer(anchor_cqa, return_tensors='pt')['input_ids']
        anchor_cqa = pad_seq(ids1, stats['max_len_cqa'])

        ids2 = tokenizer(positive_cqa, return_tensors='pt')['input_ids']
        positive_cqa = pad_seq(ids2, stats['max_len_cqa'])

        anchor_entities, anchor_entities_index = pad_entities(tokenizer, anchor_graph[0], stats['max_feature_len'])
        positive_entities, positive_entities_index = pad_entities(tokenizer, positive_graph[0],
                                                                  stats['max_feature_len'])

        anchor_properties, anchor_properties_index = pad_entities(tokenizer, anchor_graph[1], stats['max_property_len'])
        positive_properties, positive_properties_index = pad_entities(tokenizer, positive_graph[1],
                                                                      stats['max_property_len'])

        edge1 = torch.LongTensor(anchor_graph[2])
        edge2 = torch.LongTensor(positive_graph[2])

        pd.append(
            PadData(cqa_name, anchor_cqa, positive_cqa, anchor_entities, anchor_entities_index, positive_entities,
                    positive_entities_index, anchor_properties, anchor_properties_index, positive_properties,
                    positive_properties_index, edge1, edge2))

    return pd


class GraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):

        if key == 'rsi':
            return self.x_s.size(0)
        if key == 'rpi':
            return self.x_p.size(0)
        if key == 'rni':
            return self.x_n.size(0)

        if key == 'x_s':
            return self.x_sf.size(0)
        if key == 'x_p':
            return self.x_pf.size(0)
        if key == 'x_n':
            return self.x_nf.size(0)

        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_p':
            return self.x_p.size(0)
        if key == 'edge_index_n':
            return self.x_n.size(0)

        if key == 'edge_feat_s':
            return self.edge_feat_sf.size(0)

        if key == 'edge_feat_p':
            return self.edge_feat_pf.size(0)

        if key == 'edge_feat_n':
            return self.edge_feat_nf.size(0)

        return super().__inc__(key, value, *args, **kwargs)


def build_graph_dataset(tokenizer, cqas, idata, raw_data):
    stats = get_dataset_stats(raw_data)
    pd = pad_dataset(tokenizer, raw_data, stats)
    dataset = []

    # for cqa, c1, c2, f1, fi1, f2, fi2, p1, pi1, p2, pi2, e1, e2 in pd:
    for data in pd:
        rn = random.choice(list(cqas.keys()))
        rcq = random.choice(list(set(cqas[rn].keys()) - {data.anchor_cqa}))
        rcqa = cqas[rn][rcq]

        ids1 = tokenizer(rcqa, return_tensors='pt')['input_ids']
        negative_cqa = pad_seq(ids1, stats['max_len_cqa'])

        re = random.choice(list(set(idata.keys())))
        rce = random.choice(list(set(idata[re].keys()) - {data.anchor_cqa}))
        rcea = idata[re][rce]

        negative_entity, negative_entity_index = pad_entities(tokenizer, rcea[0], stats['max_feature_len'])

        negative_property, negative_property_index = pad_entities(tokenizer, rcea[1], stats['max_property_len'])

        edge1 = torch.LongTensor(rcea[2])

        dataset.append(GraphData(
            rsi=torch.LongTensor([0]),
            rpi=torch.LongTensor([0]),
            rni=torch.LongTensor([0]),
            cqs=data.anchor_cqa.long(),
            cqp=data.positive_cqa.long(),
            cqn=negative_cqa.long(),
            x_s=data.anchor_entities_index.long(),
            x_sf=data.anchor_entities.long(),
            x_p=data.positive_entities_index.long(),
            x_pf=data.positive_entities.long(),
            x_n=negative_entity_index.long(),
            x_nf=negative_entity.long(),
            edge_index_s=data.edge1.long(),
            edge_index_p=data.edge2.long(),
            edge_index_n=edge1.long(),
            edge_feat_s=data.anchor_properties_index.long(),
            edge_feat_sf=data.anchor_properties.long(),
            edge_feat_p=data.positive_properties_index.long(),
            edge_feat_pf=data.positive_properties.long(),
            edge_feat_n=negative_property_index.long(),
            edge_feat_nf=negative_property.long(),
        ))

    return dataset
