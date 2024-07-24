import os
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import gc
from multiprocessing_on_dill import Process

base_path = '/projets/melodi/gsantoss/complex-llm/generated-prompts'
base_out = '/projets/melodi/gsantoss/complex-llm/generated-edoal'


def match(txt, tokenizer, model):
    messages = [{"role": "user", "content": txt},]

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
        with open(l) as f:
            txt = f.read()
            file_out = l.replace('generated-prompts', 'generated-edoal').replace('.txt', '.edoal')
            fl = '/'.join(file_out.split('/')[:-1])

            os.makedirs(fl, exist_ok=True)
            with open(fl, 'w') as fl:
                fl.write(match(txt, tokenizer, model))

            gc.collect()
            torch.cuda.empty_cache()

        print(f'Process {i} - {ci} / {len(lines)} : {ci / len(lines) * 100:.2f}%')


if __name__ == '__main__':

    lines = []

    for p, d, fs in os.walk(base_path):
        for f in fs:
            fp = os.path.join(p, f)
            if 'conference_100' in fp or 'hydrogra' in fp or 'nq-s1-ns2' in fp:
                continue

            lines.append(fp)

    poll = []

    slice_size = len(lines) // 4
    for i in range(0, 4):
        slines = lines[i * slice_size: (i + 1) * slice_size]
        poll.append(Process(target=run, args=(i, slines,)))

    for p in poll:
        p.start()
        p.join()
