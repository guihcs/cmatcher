import os
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import gc
from multiprocessing_on_dill import Process
import sys

base_path = '/projets/melodi/gsantoss/complex-llm/generated-prompts'
base_out = '/projets/melodi/gsantoss/complex-llm/generated-edoal'


def match(txt, tokenizer, model):
    messages = [{"role": "user", "content": txt}, ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=2 * 1024,
            eos_token_id=terminators,
            do_sample=False,
            temperature=None,
            top_p=None,

        )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)


def run(i, lines):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=i,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True

    )
    model.eval()

    for ci, l in enumerate(lines):
        file_out = l.replace('generated-prompts', 'generated-edoal').replace('.txt', '.edoal')

        if os.path.exists(file_out):
            with open(file_out) as f:
                if len(f.read()) > 0:
                    print('skipping ', ci)
                    continue

        fl = '/'.join(file_out.split('/')[:-1])

        os.makedirs(fl, exist_ok=True)

        with open(l) as f:
            txt = f.read()

        try:
            with open(file_out, 'w') as fl:
                fl.write(match(txt, tokenizer, model))
        except Exception as e:
            print(f'Error in {ci}: {e}')
            continue

        gc.collect()
        torch.cuda.empty_cache()

        print(f'Process {i} - {ci} / {len(lines)} : {ci / len(lines) * 100:.2f}%')


if __name__ == '__main__':

    block = int(sys.argv[1])
    block_count = int(sys.argv[2])
    lines = []

    for p, d, fs in os.walk(base_path):
        for f in fs:
            fp = os.path.join(p, f)
            if 'conference_100' in fp or 'hydrogra' in fp or 'nq-s1-ns2' in fp:
                continue

            lines.append(fp)

    lines.sort()

    poll = []

    slice_size = len(lines) // block_count

    if block >= block_count - 1:
        slines = lines[block * slice_size:]
    else:
        slines = lines[block * slice_size: (block + 1) * slice_size]

    run(block, slines)
