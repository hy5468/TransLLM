# TransLLM: Why Not Transform Chat Large Language Models to Non-English?

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

TransLLM is implemented based on the Chinese-LLaMA-Alpaca-2 project.

## Data
We provide the following data:
- Recovery KD data in English: ./code/train/distil_alpaca_en_52k_llama2-7b-chat.json
- Recovery KD data in Thai: ./code/train/distil_alpaca_en_52k_llama2-7b-chat_th_googlemt.json
- Alpaca-GPT-4 data in English: ./code/train/alpaca_gpt4_data_en.json
- Alpaca-GPT-4 data in Thai: ./code/train/alpaca_gpt4_data_th_googlemt.json
- MT-Bench in Thai: ./code/test/mt_bench_question.xlsx
- Alpaca-Eval in Thai: ./code/test/alpaca_eval.xlsx
- Example data format of experiments: ./code/train/example

## Traning

### Model Extension
Use SentencePiece to learn the Thai vocabulary on mc4-TH. Merege the vocabulary as described in Chinese-LLaMA-Alpaca-2.

### Target Language Pre-Training

- Prepare mc4-TH in txt format, and the target chat model (such as llama2-chat-7b-hf).
- Change the data path and model path in the ./train/run_pt_1.sh.
- Run run_pt_1.sh.

### Translation Pre-Training
- Prepare Pile data and EN-TH parallel data in txt format
- Change the data path and model path in the ./train/run_pt_2.sh.
- Run run_pt_2.sh.

### Transfer Fine-Tuning
- Translate the Recovery KD data to Thai, organize TCOT data and SFT Translation data.
- Change the data path and model path in the ./train/run_sft.sh.
- Run run_sft.sh.

## Evluation

We provide the following scripts for evaluation
- Merge the LoRA model: ./Chinese-LLaMA-Alpaca-2/scripts/merge_llama2_with_chinese_lora_low_mem.py
- Generate output for mt_bench: ./eval/mt_bench_generate.py
- Generate output for alpaca_eval: ./eval/alpaca_eval_generate.py
- Generate GPT-4 evaluations: ./eval/gpt4_eval.py

## Notice
We have modified some files in ./Chinese-LLaMA-Alpaca-2/scripts/training.
- run_clm_pt_with_peft.py
- run_clm_sft_with_peft.py
- build_dataset.py
- build_distil_dataset.py

## License
The code and data is released under Apache License 2.0.

## Citation
Please cite as:
``` bibtex
@misc{geng2024TransLLM,
      title={Why Not Transform Chat Large Language Models to Non-English?}, 
      author={Xiang Geng and Ming Zhu and Jiahuan Li and Zhejian Lai and Wei Zou and Shuaijie She and Jiaxin Guo and Xiaofeng Zhao and Yinglu Li and Yuang Li and Chang Su and Yanqing Zhao and Min Zhang and Hao Yang and Xinglin Lyu and Jiajun Chen and Shujian Huang},
      year={2024},
      eprint={2405.13923},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

