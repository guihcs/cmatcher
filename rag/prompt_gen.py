import os
import itertools
import sys

from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from rdflib import Graph
from rag_reduce import ont_query_reduce


def gen_prompt(r1, r2, query, sample1='', sample2=''):
    sample_prompt = 'And examples of complex alignment between different ontologies:'

    instruction = "Write a file in EDOAL format containing the complex alignment between the input ontologies <ontology1> and <ontology2>. You don't need to explain yourself. Just give as response the resulting alignment file without saying anything else."

    if query is not None:
        instruction = f'Considering that the input ontologies were filtered to include only the entities related to the query:\n\n{query}\n\n{instruction}'

    if sample1 == '' and sample2 == '':
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
