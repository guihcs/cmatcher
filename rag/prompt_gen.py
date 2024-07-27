import os
import itertools
import sys

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from rdflib import Graph
from rag_reduce import ont_query_reduce

paths = {
    'conference': '/projets/melodi/gsantoss/data/complex/conference/ont',
    'geolink': '/projets/melodi/gsantoss/data/complex/geolink'
}

cqp = '/projets/melodi/gsantoss/data/cqas/prcqas'
cqas = {}

for p, d, fs in os.walk(cqp):
    for f in fs:
        cq = '/'.join(p.split('/')[-2:])
        if cq not in cqas:
            cqas[cq] = []

        cqas[cq].append(os.path.join(p, f))

ignore = {'paperdyne.owl', 'PCS.owl', 'Cocus.owl', 'confious.owl', 'crs_dr.owl', 'linklings.owl', 'MyReview.owl',
          'OpenConf.owl', 'MICRO.owl', '.DS_Store'}

params = []

for name, path in paths.items():
    onts = set()
    for p, d, fs in os.walk(path):
        for f in fs:
            onts.add(f)

    res = onts - ignore

    for o1, o2 in itertools.permutations(res, 2):
        k = f'{name}/{o1.split(".")[0]}'
        if k not in cqas:
            continue

        for q in cqas[k]:
            params.append((q, os.path.join(path, o1), os.path.join(path, o2)))

params.sort(key=lambda x: '.'.join(x))

if sys.argv[1] == 'count':
    print(len(params))
    sys.exit()

run_num = int(sys.argv[1])

run = params[run_num]

print('Loading model and tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-2_R')
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModel.from_pretrained(
    'Salesforce/SFR-Embedding-2_R',
    quantization_config=quantization_config,
    device_map='auto',
)
model.eval()

print('Model and tokenizer loaded successfully!')

prompt = 'Given the following SPARQL query, retrieve relevant entities that are related to the query'

g1 = Graph().parse(run[1])
g2 = Graph().parse(run[2])

with open(run[0], 'r') as f:
    query = f.read()

r1 = ont_query_reduce(model, tokenizer, run[1], g1, query, prompt, max_entities=10, batch_size=2)
r2 = ont_query_reduce(model, tokenizer, run[2], g2, query, prompt, max_entities=10, batch_size=2)


sample1_path = sys.argv[2]
sample2_path = sys.argv[3]

def gen_prompt(r1, r2, query, include_sample1=False, include_sample2=False):
    sample_prompt = 'And examples of complex alignment between different ontologies:'
    with open(sample1_path, 'r') as f:
        sample1 = f.read()

    with open(sample2_path, 'r') as f:
        sample2 = f.read()

    instruction = "Write a file in EDOAL format containing the complex alignment between the input ontologies <ontology1> and <ontology2>. You don't need to explain yourself. Just give as response the resulting alignment file without saying anything else."

    if query is not None:
        instruction = f'Considering that the input ontologies were filtered to include only the entities related to the query:\n\n{query}\n\n{instruction}'

    if not include_sample1:
        sample1 = ''

    if not include_sample2:
        sample2 = ''

    if not include_sample1 and not include_sample2:
        sample_prompt = ''

    return f'''Given the two ontologies below:
<ontology1>
{r1}    
</ontology1>    
<ontology2>
{r2}
</ontology2>
{sample_prompt}
{sample1}
{sample2}
{instruction}'''


out = '/projets/melodi/gsantoss/complex-llm/generated-prompts'

cqn = run[0].split('/')[-1].split('.')[0]
fn1 = run[1].removeprefix('/projets/melodi/gsantoss/data/complex/')
fn2 = run[2].removeprefix('/projets/melodi/gsantoss/data/complex/')
fp = fn1.split('/')
fn1 = fp[-1].split('.')[0]
fp = '/'.join(fp[:-1])
fn2 = fn2.split('/')[-1].split('.')[0]

os.makedirs(f'{out}/{fp}', exist_ok=True)

with open(f'{out}/{fp}/prompt#{fn1}#{fn2}#{cqn}#nq-ns1-ns2.txt', 'w') as f:
    f.write(gen_prompt(r1, r2, None, include_sample1=False, include_sample2=False))

with open(f'{out}/{fp}/prompt#{fn1}#{fn2}#{cqn}#nq-s1-s2.txt', 'w') as f:
    f.write(gen_prompt(r1, r2, None, include_sample1=True, include_sample2=True))

with open(f'{out}/{fp}/prompt#{fn1}#{fn2}#{cqn}#q-s1-s2.txt', 'w') as f:
    f.write(gen_prompt(r1, r2, query, include_sample1=True, include_sample2=True))
