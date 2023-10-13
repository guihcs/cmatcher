import itertools
import torch
import random
from torch_geometric.data import Data

def build_raw_data(idata, cqas):

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
            'max_property_count': max_property_count, 'max_property_len': max_property_len, 'max_edge_count': max_edge_count}


def pad_seq(t, max_len):
    return torch.cat([t, torch.zeros((t.shape[0], max_len - t.shape[1]))], dim=1)


def pad_seq2(t, max_len):
    return torch.cat([t, torch.zeros((max_len - t.shape[0], t.shape[1]))], dim=0)


def pad_edge(t, max_len):
    return t + (max_len - len(t)) * [[-1, -1]]


def pad_dataset(tokenizer, raw_data, stats):
    pd = []
    for cqa, c1, c2, e1, e2 in raw_data:
        ids1 = tokenizer(c1, return_tensors='pt')['input_ids']
        ids2 = tokenizer(c2, return_tensors='pt')['input_ids']
        nids1 = pad_seq(ids1, stats['max_len_cqa'])
        nids2 = pad_seq(ids2, stats['max_len_cqa'])

        e1id = tokenizer(e1[0], return_tensors='pt', padding=True)['input_ids']
        e2id = tokenizer(e2[0], return_tensors='pt', padding=True)['input_ids']
        pd1 = pad_seq(e1id, stats['max_feature_len'])
        pd1 = torch.cat([torch.zeros((1, stats['max_feature_len'])), pd1], dim=0)
        pd2 = pad_seq(e2id, stats['max_feature_len'])
        pd2 = torch.cat([torch.zeros((1, stats['max_feature_len'])), pd2], dim=0)

        e1pid = tokenizer(e1[1], return_tensors='pt', padding=True)['input_ids']
        e2pid = tokenizer(e2[1], return_tensors='pt', padding=True)['input_ids']
        pd3 = pad_seq(e1pid, stats['max_property_len'])
        pd4 = pad_seq(e2pid, stats['max_property_len'])

        edge1 = torch.LongTensor(e1[2])
        edge2 = torch.LongTensor(e2[2])

        pd.append((cqa, nids1, nids2, pd1, pd2, pd3, pd4, edge1, edge2))

    return pd


class GraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cqs':
            return self.x_s.size(0)
        if key == 'cqp':
            return self.x_p.size(0)
        if key == 'cqn':
            return self.x_n.size(0)
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_p':
            return self.x_p.size(0)
        if key == 'edge_index_n':
            return self.x_n.size(0)
        if key == 'edge_feat_s':
            return self.x_s.size(0)
        if key == 'edge_feat_p':
            return self.x_p.size(0)
        if key == 'edge_feat_n':
            return self.x_n.size(0)
        if key == 'rsi':
            return self.x_s.size(0)
        if key == 'rpi':
            return self.x_p.size(0)
        if key == 'rni':
            return self.x_n.size(0)

        return super().__inc__(key, value, *args, **kwargs)

def build_graph_dataset(tokenizer, cqas, idata, raw_data):
    stats = get_dataset_stats(raw_data)
    pd = pad_dataset(tokenizer, raw_data, stats)
    dataset = []

    for cqa, c1, c2, f1, f2, p1, p2, e1, e2 in pd:
        rn = random.choice(list(cqas.keys()))
        rcq = random.choice(list(set(cqas[rn].keys()) - {cqa}))
        rcqa = cqas[rn][rcq]

        re = random.choice(list(set(idata.keys())))
        rce = random.choice(list(set(idata[re].keys()) - {cqa}))
        rcea = idata[re][rce]

        ids1 = tokenizer(rcqa, return_tensors='pt')['input_ids']
        pids1 = pad_seq(ids1, stats['max_len_cqa'])

        e1id = tokenizer(rcea[0], return_tensors='pt', padding=True)['input_ids']
        pd1 = pad_seq(e1id, stats['max_feature_len'])
        pd1 = torch.cat([torch.zeros((1, stats['max_feature_len'])), pd1], dim=0)
        e1pid = tokenizer(rcea[1], return_tensors='pt', padding=True)['input_ids']
        pd3 = pad_seq(e1pid, stats['max_property_len'])

        edge1 = torch.LongTensor(rcea[2])

        dataset.append(GraphData(
            rsi=torch.LongTensor([1]),
            rpi=torch.LongTensor([1]),
            rni=torch.LongTensor([1]),
            cqs=c1,
            cqp=c2,
            cqn=pids1,
            x_s=f1,
            x_p=f2,
            x_n=pd1,
            edge_index_s=e1,
            edge_index_p=e1,
            edge_index_n=edge1,
            edge_feat_s=p1,
            edge_feat_p=p2,
            edge_feat_n=pd3,
        ))

    return dataset


