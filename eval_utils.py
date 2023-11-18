from tqdm.auto import tqdm

import torch
from cmatcher.cqa_search import pad_entities, GraphData

def prepare_eval_dataset(data, ifd, tokenizer, mc, mp):
    ts = []
    graph_data = []
    for s, cm, pm, fm in tqdm(ifd):
        pd1, pdi1 = pad_entities(tokenizer, cm, mc)

        pd3, pdi3 = pad_entities(tokenizer, pm, mp)

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
    for k in data:
        cq.append(k)
        cqi.append(data[k])

    cqid = tokenizer(cqi, return_tensors='pt', padding=True)['input_ids']

    return ts, graph_data, cq, cqid


def embed_subg(accelerator, model, graph_loader):

    fe = []
    for batch in graph_loader:
        with torch.no_grad():

            _, out, _ = model(positive_sbg=(batch.x_sf, batch.x_s, batch.edge_index_s,
                                            batch.edge_feat_sf, batch.edge_feat_s))
            fe.append(accelerator.gather_for_metrics(out[batch.rsi]))

    fe = torch.cat(fe, dim=0)

    return fe


def embed_cqas(accelerator, model, cqloader):

    cqeb = []

    for c in cqloader:
        with torch.no_grad():

            out, _, _ = model(cqa=c)
            cqeb.append(accelerator.gather_for_metrics(out))

    cqeb = torch.cat(cqeb, dim=0)
    return cqeb


def eval_metrics(cqa_list, cqa_embeddings, fe, ts, res, th=0.8):
    metrics = []

    for cqa_name, e in zip(cqa_list, cqa_embeddings):

        sim = torch.cosine_similarity(e.unsqueeze(0), fe, dim=1)
        resid = torch.where(sim > th)[0].tolist()
        rs = set()
        for r in resid:
            rs.add(ts[r])

        metrics.append((1 if res[cqa_name] in rs else 0, len(rs)))

    avgp = sum([1 / m[1] if m[1] > 0 else 0 for m in metrics]) / len(metrics)
    rc = sum([m[0] for m in metrics]) / len(metrics)
    fm = 2 * rc * avgp / (rc + avgp) if rc + avgp > 0 else 0
    return avgp, rc, fm