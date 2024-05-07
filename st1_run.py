# %%
import os

os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ["WANDB_DIR"] = '/projets/melodi/gsantoss/wandbt'

from owl_utils import *
from cqa_search import *
from eval_utils import *
from transformers import AutoTokenizer
import dill
import torch
import torch.optim as optim
import copy
import tqdm
from model import *
from tqdm.auto import tqdm
import random
import wandb

import argparse

torch.manual_seed(0)
random.seed(0)




def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='')

    arg_parser.add_argument('--sweep', dest='sweep', nargs='?', type=int)

    return arg_parser.parse_args()


# %%


args = parse_arguments()

test_onts = ['cmt', 'conference', 'confOf', 'edas', 'ekaw']
language_models = ['BAAI/bge-base-en', 'infgrad/stella-base-en-v2', 'BAAI/bge-large-en-v1.5', 'llmrails/ember-v1',
                   'thenlper/gte-large']
architectures = ['lm', 'gnn', 'sgnn']
lm_grad = ['none', 'grad']
pred = ['none', 'pred']
dephs = [1, 2, 3, 4]


def all_combinations():
    combs = []
    for to in test_onts:
        for lm in language_models:
            for a in architectures:
                if a == 'lm':
                    combs.append((to, lm, a, 'grad', 'none', 0))
                    continue
                for g in lm_grad:
                    for p in pred:
                        for d in dephs:
                            combs.append((to, lm, a, g, p, d))

    return combs


test_ont, language_model, architecture, grad, cpred, depth = all_combinations()[args.sweep]
# test_ont, language_model, architecture, grad, cpred, depth = all_combinations()[0]

config = {
    'test_ont': test_ont,
    'learning_rate': 0.00001,
    'language_model': language_model,
    'architecture': architecture,
    'pred': cpred,
    'epochs': 5,
    'batch_size': 2,
    'evm_th': 0.9,
    'ev_sim_threshold': 0.8,
    'sim_margin': 0.8,
    'depth': depth,
    'grad': grad
}

ontology_paths = {
    'edas.owl': '/projets/melodi/gsantoss/data/oaei/tracks/conference/onts/edas.owl',
    'ekaw.owl': '/projets/melodi/gsantoss/data/oaei/tracks/conference/onts/ekaw.owl',
    'confOf.owl': '/projets/melodi/gsantoss/data/oaei/tracks/conference/onts/confOf.owl',
    'conference.owl': '/projets/melodi/gsantoss/data/oaei/tracks/conference/onts/conference.owl',
    'cmt.owl': '/projets/melodi/gsantoss/data/oaei/tracks/conference/onts/cmt.owl',
}

cqa_path = '/projets/melodi/gsantoss/data/complex/CQAs'
entities_path = '/projets/melodi/gsantoss/data/complex/entities-cqas'

if os.path.exists('/projets/melodi/gsantoss/tmp/idata.pkl'):
    with open('/projets/melodi/gsantoss/tmp/idata.pkl', 'rb') as f:
        train_ont_cqa_subg = dill.load(f)
        print('loaded from cache.')
else:
    with open('/projets/melodi/gsantoss/tmp/idata.pkl', 'wb') as f:
        dill.dump(load_entities(entities_path, ontology_paths), f)

isg = load_sg(entities_path, ontology_paths)

cqas = load_cqas(cqa_path)
raw_data = build_raw_data(train_ont_cqa_subg, cqas)

test_ont = config['test_ont']

if os.path.exists(f'/projets/melodi/gsantoss/tmp/{test_ont}.pkl'):
    with open(f'/projets/melodi/gsantoss/tmp/{test_ont}.pkl', 'rb') as f:
        ifd, mc, mp, fres = dill.load(f)
        print('loaded from cache.')
else:
    ifd, mc, mp, fres = build_raw_ts(f'/projets/melodi/gsantoss/data/oaei/tracks/conference/onts/{test_ont}.owl',
                                     isg[test_ont],
                                     workers=4)
    with open(f'/projets/melodi/gsantoss/tmp/{test_ont}.pkl', 'wb') as f:
        dill.dump((ifd, mc, mp, fres), f)

conts_cqa_subg = copy.deepcopy(train_ont_cqa_subg)
del conts_cqa_subg[test_ont]

tokenizer = AutoTokenizer.from_pretrained(config['language_model'])

root_entities, graph_data, cq, cqid, caq, cqmask, tor = prepare_eval_dataset(test_ont, cqas, ifd, tokenizer, mc, mp,
                                                                             fres)

wandb.init(
    project='cmatcher',
    config=config,
    group=f'{language_model}-{architecture}-{cpred}-{grad}',
    settings=wandb.Settings(_disable_stats=True, _disable_meta=True)
)

print(config)

# %%
print('start training')

# %%
model = Model(config['language_model'], d=config['depth'], lm_grad=config['grad'] == 'grad')
model.cuda(0)


# %%


def evm(model, dataset, th=0.5):
    model.eval()

    res = []
    print('begin evm')
    for batch in DataLoader(dataset, batch_size=2):
        with torch.no_grad():
            cqs, sbgs, _ = model(cqa=batch.cqs.cuda(0), positive_sbg=(batch.x_sf.cuda(0), batch.x_s.cuda(0),
                                                                      batch.edge_index_s.cuda(0),
                                                                      batch.edge_feat_sf.cuda(0),
                                                                      batch.edge_feat_s.cuda(0)))

            isbgs = sbgs[batch.rsi]

            sim = torch.cosine_similarity(cqs, isbgs) > th
            res.append(sim)

    res = torch.cat(res, dim=0)
    print('end evm')
    return (res.sum() / res.size(0)).item()


# model = Model(config['language_model'], d=config['depth'], lm_grad=config['grad'] == 'grad')

# %%
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

lh = []
evh = []
epochs = config['epochs']
batch_size = config['batch_size']
progress = None

triplet_loss = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda x, y: 1.0 - torch.cosine_similarity(x, y), margin=config['sim_margin'])

print('build datasets')
dataset = CQADataset(tokenizer, conts_cqa_subg, raw_data[test_ont], filter_bn=False)
loader = DataLoader(dataset, batch_size=batch_size)

cqloader = DataLoader(cqid, batch_size=batch_size, shuffle=False)
acqloader = [DataLoader(a, batch_size=batch_size, shuffle=False) for a in caq]
graph_loader = DataLoader(graph_data, batch_size=batch_size, shuffle=False)
# %%


print('data prepared')
model.find_unused_parameters = False
if not progress:
    progress = tqdm(total=epochs * len(loader))

print('start training')
evh.append(evm(model, dataset, th=config["ev_sim_threshold"]))
eval_test(model, cqloader, graph_loader, cq, root_entities, fres, acqloader, cqmask, tor)
wandb.log({'global/acc': evh[-1]})

for e in range(epochs):

    model.train()

    el = []
    for batch in loader:
        optimizer.zero_grad()

        cqs, sbgs, nsbg = model(cqa=batch.cqs.cuda(0), positive_sbg=(batch.x_sf.cuda(0), batch.x_s.cuda(0),
                                                                     batch.edge_index_s.cuda(0),
                                                                     batch.edge_feat_sf.cuda(0),
                                                                     batch.edge_feat_s.cuda(0)),
                                negative_sbg=(batch.x_nf.cuda(0), batch.x_n.cuda(0),
                                              batch.edge_index_n.cuda(0), batch.edge_feat_nf.cuda(0),
                                              batch.edge_feat_n.cuda(0)))

        isbgs = sbgs[batch.rsi]
        isbgn = nsbg[batch.rni]

        loss = triplet_loss(cqs, isbgs, isbgn)
        el.append(loss.detach())
        loss.backward()

        optimizer.step()
        progress.update(1)

    lh.append(torch.stack(el).mean().item())

    evh.append(evm(model, dataset, th=config["ev_sim_threshold"]))
    eval_test(model, cqloader, graph_loader, cq, root_entities, fres, acqloader, cqmask, tor)
    wandb.log({'global/acc': evh[-1], 'global/loss': lh[-1]})

progress.close()

wandb.finish()

# %%
