from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
import torch
import json

alpaca_eval_path = "xxx.jsonl"
alpaca_eval_output_path = "xxx.jsonl"
model_path = "model_path/"
f = open(alpaca_eval_path, 'r', encoding='utf8')
f_out = open(alpaca_eval_output_path, 'w', encoding='utf8')
device = torch.device(0)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=0)
tokenizer = LlamaTokenizer.from_pretrained(model_path)
template = ("<s>[INST] \n\n<<SYS>>You are a helpful<</SYS>>\n\n{instruct}[/INST]")

for line in f:
    data = json.loads(line)
    prompt = data['instruction']
    input_text = tokenizer(template.format_map({"instruct":prompt}), return_tensors="pt", add_special_tokens=False)
    response = model.generate(**input_text,eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    num_beams=1, do_sample=False
                                )
    output = tokenizer.batch_decoded(response)[0]
    th_output = output.split("<TH>")[1].replace("</s>", "").strip()
    f_out.write(json.dumps({"instruction": prompt, "output": th_output}, ensure_ascii=False)+'\n')