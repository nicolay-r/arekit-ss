## arekit-ss 0.23.1

![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.23.1-orange.svg)
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/arekit-ss/blob/master/arekit_ss.ipynb)

<p align="center">
    <img src="logo.png"/>
</p>

`arekit-ss` [AREkit double "s"] -- is an extension for instant object-pair context sampling from 
[AREkit](https://github.com/nicolay-r/AREkit)
collection of 
[datasources](https://github.com/nicolay-r/AREkit/wiki/Binded-Sources).

For custom text sampling, please follow the 
[ARElight](https://github.com/nicolay-r/ARElight)
project.

> **Mind the case (issue [#18](https://github.com/nicolay-r/arekit-ss/issues/18)):** switching to another language may changed amount of extracted data due to `terms_per_context` parameter
that crops context with fixed amount of words.

## Installation

Install dependencies:
```bash
pip install git+https://github.com/nicolay-r/arekit-ss.git@master
```

Download AREkit related data, from which `sources` are required:
```bash
python -m arekit.download_data
```

## Usage
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/arekit-ss/blob/master/arekit_ss.ipynb)

Example of composing prompts:
```bash
python -m arekit_ss.sample --writer csv --source rusentrel --sampler prompt \
  --prompt "For text: '{text}', the attitude between '{s_val}' and '{t_val}' is: '{label_val}'" \
  --dest_lang en --docs_limit 1 --text_parser lm --output_dir 'out_rusentrel_prompt'
```
![output_prompts](https://github.com/nicolay-r/arekit-ss/assets/14871187/d1499f24-b2df-410b-98cc-8d4018de8c65)

### samplers
* `nn` -- CNN/LSTM architecture related, including frames annotation from [RuAttitudes](https://github.com/nicolay-r/RuAttitudes).
* `bert` -- BERT-based, single-input sequence.  
* `prompt` -- prompt-based sampler for `ChatGPT` and the related conversational systems [[prompt engeneering guide]](https://github.com/dair-ai/Prompt-Engineering-Guide)

### Writers
* `csv` -- for [AREnets](https://github.com/nicolay-r/AREnets) framework;
* `jsonl` -- for [OpenNRE](https://github.com/thunlp/OpenNRE) framework.

## Powered by

* [AREkit framework](https://github.com/nicolay-r/AREkit)
