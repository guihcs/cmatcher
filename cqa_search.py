import itertools
import torch
import random
from torch_geometric.data import Data, Dataset
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


def pad_seq(t, max_len, pad_token=0):
    return torch.cat([t, torch.ones((t.shape[0], max_len - t.shape[1]), dtype=torch.long) * pad_token], dim=1)


def pad_edge(t, max_len):
    return t + (max_len - len(t)) * [[-1, -1]]


def pad_entities(tokenizer, entities, ml, flat_bn=True):
    if flat_bn:
        ft = list(map(lambda x: x if type(x) is not BNode else 'blank node', entities))
    else:
        ft = entities

    ft = list(map(str, ft))
    sm = {}
    n = []
    for i, f in enumerate(ft):
        if f not in sm:
            sm[f] = len(sm)
            n.append(f)

        ft[i] = sm[f]
    e1id = tokenizer(n, return_tensors='pt', padding=True)['input_ids']
    return pad_seq(e1id, ml, pad_token=tokenizer.pad_token_id), torch.LongTensor(ft)


class PadData:

    def __init__(self, cqa_name, anchor_cqa, anchor_entities, anchor_entities_index,
                 anchor_properties, anchor_properties_index, edge1):
        self.cqa_name = cqa_name
        self.anchor_cqa = anchor_cqa
        self.anchor_entities = anchor_entities
        self.anchor_entities_index = anchor_entities_index
        self.anchor_properties = anchor_properties
        self.anchor_properties_index = anchor_properties_index
        self.edge1 = edge1


def pad_dataset(tokenizer, raw_data, stats, flat_bn=True) -> list[PadData]:
    pd = []
    for cqa_name, anchor_cqa, positive_cqa, anchor_graph, positive_graph in raw_data:
        ids1 = tokenizer(anchor_cqa, return_tensors='pt')['input_ids']
        anchor_cqa = pad_seq(ids1, stats['max_len_cqa'], pad_token=tokenizer.pad_token_id)

        anchor_entities, anchor_entities_index = pad_entities(tokenizer, anchor_graph[0], stats['max_feature_len'],
                                                              flat_bn=flat_bn)

        anchor_properties, anchor_properties_index = pad_entities(tokenizer, anchor_graph[1], stats['max_property_len'],
                                                                  flat_bn=flat_bn)

        edge1 = torch.LongTensor(anchor_graph[2])

        pd.append(
            PadData(cqa_name, anchor_cqa, anchor_entities, anchor_entities_index,
                    anchor_properties, anchor_properties_index,
                    edge1))

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


class CQADataset(Dataset):
    def __init__(self, tokenizer, idata, raw_data, transform=None, pre_transform=None, filter_bn=False, flat_bn=True):
        super().__init__(None, transform, pre_transform)
        self.flat_bn = flat_bn
        self.stats = get_dataset_stats(raw_data)
        self.pd = pad_dataset(tokenizer, raw_data, self.stats, flat_bn=flat_bn)
        self.idata = idata
        self.tokenizer = tokenizer

        if filter_bn:
            self.pd = list(filter(lambda x: 'blank node' not in tokenizer.decode(x.anchor_entities[0]), self.pd))

    def len(self):
        return len(self.pd)

    def get(self, idx):
        data = self.pd[idx]
        re = random.choice(list(set(self.idata.keys())))
        rce = random.choice(list(set(self.idata[re].keys()) - {data.anchor_cqa}))
        rcea = self.idata[re][rce]

        negative_entity, negative_entity_index = pad_entities(self.tokenizer, rcea[0], self.stats['max_feature_len'],
                                                              flat_bn=self.flat_bn)

        negative_property, negative_property_index = pad_entities(self.tokenizer, rcea[1],
                                                                  self.stats['max_property_len'], flat_bn=self.flat_bn)

        edge1 = torch.LongTensor(rcea[2])

        return GraphData(
            rsi=torch.LongTensor([0]),
            rni=torch.LongTensor([0]),
            cqs=data.anchor_cqa.long(),
            x_s=data.anchor_entities_index.long(),
            x_sf=data.anchor_entities.long(),
            x_n=negative_entity_index.long(),
            x_nf=negative_entity.long(),
            edge_index_s=data.edge1.long(),
            edge_index_n=edge1.long(),
            edge_feat_s=data.anchor_properties_index.long(),
            edge_feat_sf=data.anchor_properties.long(),
            edge_feat_n=negative_property_index.long(),
            edge_feat_nf=negative_property.long(),
        )


def prepare_eval_dataset(test_ont, cqas, ifd, tokenizer, mc, mp, fres, filter_bn=True, flat_bn=True):
    ts = []
    graph_data = []
    for s, cm, pm, fm in ifd:
        pd1, pdi1 = pad_entities(tokenizer, cm, mc, flat_bn=flat_bn)

        pd3, pdi3 = pad_entities(tokenizer, pm, mp, flat_bn=flat_bn)

        edge1 = torch.LongTensor(fm)

        ts.append(s)
        graph_data.append(GraphData(
            rsi=torch.LongTensor([0]),
            x_s=pdi1.long(),
            x_sf=pd1.long(),
            edge_index_s=edge1.long(),
            edge_feat_s=pdi3.long(),
            edge_feat_sf=pd3.long(),
        ))

    cq = []
    cqi = []

    tor = list(set(cqas.keys()) - {test_ont})
    aqi = [[] for _ in range(len(tor))]
    cqmask = []

    for k in cqas[test_ont]:
        if filter_bn and type(fres[k]) is BNode:
            fres.pop(k, None)
            continue

        cq.append(k)
        cqi.append(cqas[test_ont][k])

        ml = []
        for i, t in enumerate(tor):
            if k in cqas[t]:
                ml.append(1)
                aqi[i].append(cqas[t][k])
            else:
                ml.append(0)
                aqi[i].append('')

        cqmask.append(ml)

    cqid = tokenizer(cqi, return_tensors='pt', padding=True)['input_ids']

    caq = [tokenizer(a, return_tensors='pt', padding=True)['input_ids'] for a in aqi]

    return ts, graph_data, cq, cqid, caq, cqmask, tor
