import os
import itertools
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from rdflib import Graph
from rag_reduce import ont_query_reduce

paths = {
    'conference': '/projets/melodi/gsantoss/data/complex/conference/ont',
    'populated_conference': '/projets/melodi/gsantoss/data/complex/conference_100/ont',
    'geolink': '/projets/melodi/gsantoss/data/complex/geolink',
    'hydrography': '/projets/melodi/gsantoss/data/complex/hydrography_ontology/ontology',
    'taxon': '/projets/melodi/gsantoss/data/complex/taxon/ont'
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
run_num = 0

run = params[run_num]

g1 = Graph().parse(run[1])
g2 = Graph().parse(run[2])

with open(run[0], 'r') as f:
    query = f.read()
    
r1 = ont_query_reduce(model, tokenizer, g1, query, prompt, max_entities=10, batch_size=2)
r2 = ont_query_reduce(model, tokenizer, g2, query, prompt, max_entities=10, batch_size=2)

def gen_prompt(r1, r2, query, include_sample1=False, include_sample2=False):
    sample_prompt = 'Examples of complex alignment between different ontologies:'
    sample1 = '''<ontology1>
    @prefix lib: <http://example.org/library#> .
    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix foaf: <http://xmlns.com/foaf/0.1/> .

    lib:Book1 a lib:Book ;
        dcterms:title "The Catcher in the Rye" ;
        dcterms:creator lib:Author1 ;
        lib:hasGenre "Fiction" .

    lib:Author1 a lib:Author ;
        foaf:name "J.D. Salinger" ;
        foaf:birthDate "1919-01-01" .
</ontology1>
<ontology2>
    @prefix pub: <http://example.org/publishing#> .
    @prefix dcterms: <http://purl.org/dc/terms/> .
    @prefix foaf: <http://xmlns.com/foaf/0.1/> .

    pub:Book1 a pub:Book ;
        dcterms:title "To Kill a Mockingbird" ;
        dcterms:creator pub:Author1 ;
        pub:publicationYear "1960" .

    pub:Author1 a pub:Author ;
        foaf:name "Harper Lee" ;
        pub:hasNationality "American" .
</ontology2>
<result>
    <?xml version="1.0" encoding="utf-8"?>
    <rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
             xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
             xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
             xmlns:align="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"
             xmlns:edoal="http://ns.inria.org/edoal/1.0/#">    
      <Alignment>
        <xml>yes</xml>
        <level>2EDOAL</level>
        <type>**</type>        
        <onto1>
          <Ontology rdf:about="http://example.org/library#"/>
        </onto1>
        <onto2>
          <Ontology rdf:about="http://example.org/publishing#"/>
        </onto2>    
        <map>
          <Cell>
            <entity1 rdf:resource="http://example.org/library#Book"/>
            <entity2 rdf:resource="http://example.org/publishing#Book"/>
            <relation>=</relation>
            <measure>1.0</measure>
          </Cell>
        </map>
        <map>
          <Cell>
            <entity1 rdf:resource="http://example.org/library#Author"/>
            <entity2 rdf:resource="http://example.org/publishing#Author"/>
            <relation>=</relation>
            <measure>1.0</measure>
          </Cell>
        </map>
      </Alignment>
    </rdf:RDF>
</result>    
'''

    sample2 = '''<ontology1>
    @prefix : <http://example.org/ontology1/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    : ontology a owl:Ontology ;
      rdfs:label "Paper Ontology" .

    :AcceptedPaper a owl:Class ;
      rdfs:label "Accepted Paper" ;
      rdfs:comment "A paper that has been accepted for presentation at a conference" .

    :Author a owl:Class ;
      rdfs:label "Author" ;
      rdfs:comment "A person who writes a paper" .

    :Conference a owl:Class ;
      rdfs:label "Conference" ;
      rdfs:comment "An event where papers are presented" .

    :hasAuthor a owl:ObjectProperty ;
      rdfs:label "has author" ;
      rdfs:domain :AcceptedPaper ;
      rdfs:range :Author .

    :isPresentedAt a owl:ObjectProperty ;
      rdfs:label "is presented at" ;
      rdfs:domain :AcceptedPaper ;
      rdfs:range :Conference .
</ontology1>
<ontology2>
    @prefix : <http://example.org/ontology2/> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    : ontology a owl:Ontology ;
      rdfs:label "Paper Ontology" .

    :Paper a owl:Class ;
      rdfs:label "Paper" ;
      rdfs:comment "A document submitted to a conference" .

    :Decision a owl:Class ;
      rdfs:label "Decision" ;
      rdfs:comment "A verdict on a paper" .

    :Acceptance a owl:Class ;
      rdfs:subClassOf :Decision ;
      rdfs:label "Acceptance" ;
      rdfs:comment "A positive decision on a paper" .

    :hasAcceptance a owl:ObjectProperty ;
      rdfs:label "has acceptance" ;
      rdfs:domain :Paper ;
      rdfs:range :Acceptance .

    :Author a owl:Class ;
      rdfs:label "Author" ;
      rdfs:comment "A person who writes a paper" .

    :Conference a owl:Class ;
      rdfs:label "Conference" ;
      rdfs:comment "An event where papers are presented" .

    :hasAuthor a owl:ObjectProperty ;
      rdfs:label "has author" ;
      rdfs:domain :Paper ;
      rdfs:range :Author .

    :isSubmittedTo a owl:ObjectProperty ;
      rdfs:label "is submitted to" ;
      rdfs:domain :Paper ;
      rdfs:range :Conference .
</ontology2>
<result>
    <?xml version="1.0" encoding="utf-8"?>
    <rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
           xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
           xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
           xmlns:align="http://knowledgeweb.semanticweb.org/heterogeneity/alignment#"
           xmlns:edoal="http://ns.inria.org/edoal/1.0/#">
        <Alignment>
          <xml>yes</xml>
          <level>2EDOAL</level>
          <type>**</type>
          <onto1>
            <Ontology rdf:about="http://example.org/ontology1/"/>
          </onto1>
          <onto2>
            <Ontology rdf:about="http://example.org/ontology2/"/>
          </onto2>
          <map>
            <Cell>
              <entity1 rdf:resource="http://example.org/ontology1/AcceptedPaper"/>
              <entity2>
                <edoal:Class>
                  <edoal:And>
                    <edoal:Class rdf:about="http://example.org/ontology2/Paper"/>
                    <edoal:PropertyRestriction>
                      <edoal:onProperty rdf:resource="http://example.org/ontology2/hasAcceptance"/>
                      <edoal:someValuesFrom rdf:resource="http://example.org/ontology2/Acceptance"/>
                    </edoal:PropertyRestriction>
                  </edoal:And>
                </edoal:Class>
              </entity2>
              <relation>=</relation>
              <measure>1.0</measure>
            </Cell>
          </map>
          <map>
            <Cell>
              <entity1 rdf:resource="http://example.org/ontology1/Author"/>
              <entity2 rdf:resource="http://example.org/ontology2/Author"/>
              <relation>=</relation>
              <measure>1.0</measure>
            </Cell>
          </map>
          <map>
            <Cell>
              <entity1 rdf:resource="http://example.org/ontology1/Conference"/>
              <entity2 rdf:resource="http://example.org/ontology2/Conference"/>
              <relation>=</relation>
              <measure>1.0</measure>
            </Cell>
          </map>
          <map>
            <Cell>
              <entity1 rdf:resource="http://example.org/ontology1/hasAuthor"/>
              <entity2 rdf:resource="http://example.org/ontology2/hasAuthor"/>
              <relation>=</relation>
              <measure>1.0</measure>
            </Cell>
          </map>
          <map>
            <Cell>
              <entity1 rdf:resource="http://example.org/ontology1/isPresentedAt"/>
              <entity2 rdf:resource="http://example.org/ontology2/isSubmittedTo"/>
              <relation>=</relation>
              <measure>1.0</measure>
            </Cell>
          </map>
        </Alignment>
    </rdf:RDF>
</result>'''

    instruction = "Write a file in EDOAL format containing the complex alignment between the ontology1 and ontology2. You don't need to explain yourself. Just give as response the resulting alignment file without saying anything else."

    if query is not None:
        instruction = f'Considering that the input ontologies were filtered to include only the entities related to the query:\n\n{query}\n\n{instruction}'

    if not include_sample1:
        sample1 = ''

    if not include_sample2:
        sample2 = ''

    if not include_sample1 and not include_sample2:
        sample_prompt = ''

    return f'''Given the two ontologies bellow:
<ontology1>
{r1}    
</ontology1>    
<ontology2>
{r2}
</ontology2>

{sample_prompt}
{sample1}
{sample2}

{instruction}
'''


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

with open(f'{out}/{fp}/prompt#{fn1}#{fn2}#{cqn}#nq-s1-ns2.txt', 'w') as f:
    f.write(gen_prompt(r1, r2, None, include_sample1=True, include_sample2=False))

with open(f'{out}/{fp}/prompt#{fn1}#{fn2}#{cqn}#nq-s1-s2.txt', 'w') as f:
    f.write(gen_prompt(r1, r2, None, include_sample1=True, include_sample2=True))

with open(f'{out}/{fp}/prompt#{fn1}#{fn2}#{cqn}#q-s1-s2.txt', 'w') as f:
    f.write(gen_prompt(r1, r2, query, include_sample1=True, include_sample2=True))
