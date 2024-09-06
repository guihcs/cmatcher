from torch import Tensor
import torch
from ent_to_doc import gen_doc
from rdflib.namespace import RDF, RDFS, OWL, XSD
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from rdflib import Graph
import os


def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def rag(model, tokenizer, query, prompt, g, max_entities=15, max_length=4096, batch_size=2):
    queries = [
        get_detailed_instruct(prompt, query),
    ]

    ls = list(filter(lambda x: (x, RDF.first, None) not in g, set(g.subjects())))
    passages = []
    for s in ls:
        passages.append(gen_doc(s, g, max_entities=max_entities))

    input_texts = queries + passages

    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

    dataset = TensorDataset(batch_dict['input_ids'], batch_dict['attention_mask'])

    res = []
    for i, a in tqdm(DataLoader(dataset[:1], batch_size=batch_size, shuffle=False)):
        with torch.no_grad():
            output = model(input_ids=i, attention_mask=a)
            embeddings = last_token_pool(output.last_hidden_state, a)
            res.append(embeddings)

    qe = torch.cat(res, dim=0)

    base_cache = 'projects/melodi/gsantoss/rag_cache/'

    os.makedirs(base_cache, exist_ok=True)

    res = []
    for i, a in tqdm(DataLoader(dataset[1:], batch_size=batch_size, shuffle=False)):
        with torch.no_grad():
            output = model(input_ids=i, attention_mask=a)
            embeddings = last_token_pool(output.last_hidden_state, a)
            res.append(embeddings)

    ee = torch.cat(res, dim=0)

    return ls, torch.cosine_similarity(qe.unsqueeze(1), ee.unsqueeze(0), dim=2)


def reduce_ont(ls, scores, g, top_n=5, i_max_depth=3, o_max_depth=3):
    fents = [ls[x] for x, y in sorted(enumerate(scores.tolist()[0]), key=lambda x: x[1], reverse=True)[:top_n]]

    ng = Graph()

    for e in tqdm(fents):
        traverse(e, g, ng, max_depth=i_max_depth, reverse=True)
        traverse(e, g, ng, max_depth=o_max_depth)

    return ng.serialize(format='ttl')


def traverse(e, g: Graph, ng: Graph, depth=0, max_depth=3, reverse=False):
    if depth >= max_depth:
        return

    if not reverse:

        for s, p, o in g.triples((e, None, None)):
            ng.add((s, p, o))
            traverse(o, g, ng, depth=depth + 1, max_depth=max_depth, reverse=reverse)

    else:
        for s, p, o in g.triples((None, None, e)):
            ng.add((s, p, o))
            traverse(s, g, ng, depth=depth + 1, max_depth=max_depth, reverse=reverse)


def ont_query_reduce(model, tokenizer, g, query, prompt, top_n=2, i_max_depth=1, o_max_depth=2, max_entities=15,
                     max_length=4096, batch_size=2):
    ls1, scores1 = rag(model, tokenizer, query, prompt, g, max_entities=max_entities, max_length=max_length,
                       batch_size=batch_size)
    return reduce_ont(ls1, scores1, g, top_n=top_n, i_max_depth=i_max_depth, o_max_depth=o_max_depth)
