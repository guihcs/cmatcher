import torch
from torch_geometric.loader import DataLoader
import wandb


def embed_cqas(model, cqloader):
    cqeb = []

    for c in cqloader:
        with torch.no_grad():
            out, _, _ = model(cqa=c)
            cqeb.append(out)

    cqeb = torch.cat(cqeb, dim=0)
    return cqeb


def embed_subg(model, graph_loader):
    fe = []
    for batch in graph_loader:
        with torch.no_grad():
            _, out, _ = model(positive_sbg=(batch.x_sf, batch.x_s, batch.edge_index_s,
                                            batch.edge_feat_sf, batch.edge_feat_s))
            fe.append(out[batch.rsi])

    fe = torch.cat(fe, dim=0)

    return fe


def evm(model, dataset, th=0.5):
    model.eval()

    res = []
    for batch in DataLoader(dataset, batch_size=2):
        with torch.no_grad():
            cqs, sbgs, _ = model(cqa=batch.cqs, positive_sbg=(batch.x_sf, batch.x_s,
                                                              batch.edge_index_s, batch.edge_feat_sf,
                                                              batch.edge_feat_s))

            print(sbgs.shape, batch.rsi.shape, sbgs.device, batch.rsi.device)

            isbgs = sbgs[batch.rsi.cuda(0)]

            sim = torch.cosine_similarity(cqs, isbgs) > th
            res.append(sim)

    res = torch.cat(res, dim=0)

    return (res.sum() / res.size(0)).item()


def eval_metrics(cqa_list, cqa_embeddings, fe, ts, res, th=0.8, cqm=None):
    metrics = []

    for i in range(len(cqa_list)):
        if cqm is not None and cqm[i] == 0:
            continue
        cqa_name, e = cqa_list[i], cqa_embeddings[i]
        sim = torch.cosine_similarity(e.unsqueeze(0), fe, dim=1)
        resid = torch.where(sim > th)[0].tolist()
        rs = set()
        for r in resid:
            rs.add(ts[r])

        metrics.append((1 if res[cqa_name] in rs else 0, len(rs)))

    return metrics


def get_apr(metrics):
    avgp = sum([1 / m[1] if m[1] > 0 else 0 for m in metrics]) / len(metrics)
    rc = sum([m[0] for m in metrics]) / len(metrics)
    fm = 2 * rc * avgp / (rc + avgp) if rc + avgp > 0 else 0
    return avgp, rc, fm


def eval_test(model, cqloader, graph_loader, cq, root_entities, res, caq, cqmask, tor):
    model.eval()

    cqeb = embed_cqas(model, cqloader)
    aembs = [embed_cqas(model, a) for a in caq]

    graph_embeddings = embed_subg(model, graph_loader)

    avgps = []
    rcs = []
    fms = []

    for t in torch.arange(0, 1, 0.05):
        avgp, rc, fm = get_apr(eval_metrics(cq, cqeb, graph_embeddings, root_entities, res, th=t))
        avgps.append(avgp)
        rcs.append(rc)
        fms.append(fm)

    bv = torch.tensor(fms).argmax().item()

    wandb.log({'global/bt': bv * 0.05, 'global/avgp': avgps[bv], 'global/rec': rcs[bv], 'global/afm': fms[bv]})

    gavgps = 0
    grcs = 0
    gfms = 0

    for i in range(len(tor)):
        avgps = []
        rcs = []
        fms = []
        for t in torch.arange(0, 1, 0.05):
            metrics = eval_metrics(cq, aembs[i], graph_embeddings, root_entities, res, th=t,
                                   cqm=[x[i] for x in cqmask])
            avgp, rc, fm = get_apr(metrics)
            avgps.append(avgp)
            rcs.append(rc)
            fms.append(fm)

        bv = torch.tensor(fms).argmax().item()
        wandb.log({f'each/{tor[i]}-bt': bv * 0.05, f'each/{tor[i]}-avgp': avgps[bv], f'each/{tor[i]}-rec': rcs[bv],
                   f'each/{tor[i]}-afm': fms[bv]})
        gavgps += avgps[bv]
        grcs += rcs[bv]
        gfms += fms[bv]

    gavgps /= len(tor)
    grcs /= len(tor)
    gfms /= len(tor)

    wandb.log({'global/gavgp': gavgps, 'global/grec': grcs, 'global/gafm': gfms})
