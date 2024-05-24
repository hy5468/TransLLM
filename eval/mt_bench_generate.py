from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
import torch
import json

mt_bench_path = "xxx.jsonl"
mt_bench_output_path = "xxx.jsonl"
f = open(mt_bench_path, 'r', encoding='utf8')
f_out = open(mt_bench_output_path, 'w', encoding='utf8')
device = torch.device(0)
model_path = ""
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=0)
tokenizer = LlamaTokenizer.from_pretrained(model_path)
template = "<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\n{instruct}[/INST]"
template_multi_turns = "<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\n{instruct1} [/INST] {response} </s><s>[INST] {instruct2} [/INST]"

for line in f:
    question1, question2 = json.loads(line)['turns']
    input_text1 = tokenizer(template.format_map({"instruct":question1}), return_tensors="pt", add_special_tokens=False)
    response1 = model.generate(**input_text1,eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    num_beams=1, do_sample=False
                                )
    output1 = tokenizer.batch_decoded(response1)[0]
    en_output = output1.split("<RESPONSE>")[1].split("<TH>")[0].strip()
    en_input = output1.split("<EN>")[1].split('<RESPONSE>')[0].strip()
    th_output1 = output1.split("<TH>")[1].replace("</s>", "").strip()

    input_text2 = tokenizer(template_multi_turns.format_map({"instruct1": en_input, "response":en_output, "instruct2": question2}))
    response2 = model.generate(**input_text2,eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                num_beams=1, do_sample=False
                             )
    output2 = tokenizer.batch_decoded(response2)[0]
    # Since we do not train the LLM on the multi-turn task, the LLM sometimes (<5%) does not follow the TCOT format.
    if '<RESPONSE>' not in output2:
        input_text2 = input_text2.replace('</s>').strip()+' <RESPONSE>'
        response2 = model.generate(**input_text2,eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    num_beams=1, do_sample=False
                                )
        output2 = tokenizer.batch_decoded(response2)[0]
    if '<TH>' not in output2:
        input_text2 = input_text2.replace('</s>').strip()+' <TH>'
        response2 = model.generate(**input_text2,eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    num_beams=1, do_sample=False
                                )
        output2 = tokenizer.batch_decoded(response2)[0]
    th_output2 = output2.split("<TH>")[1].replace("</s>", "").strip()

    f_out.write(json.dumps({"question1": question1, "question2": question2, "output1":th_output1, "output2": th_output2}, ensure_ascii=False)+'\n')