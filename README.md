## arekit-ss 0.24.0

![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.24.0-orange.svg)
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

## Installation

Install dependencies:
```bash
pip install git+https://github.com/nicolay-r/arekit-ss.git@0.24.0
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
  --dest_lang en --docs_limit 1
```

> **Mind the case (issue [#18](https://github.com/nicolay-r/arekit-ss/issues/18)):**
> switching to another language may affect on amount of extracted data because of `terms_per_context`
> parameter that crops context by fixed and predefined amount of words.

<details>
<summary>

## Parameters
</summary>

* `source` -- source name from the list of the [supported sources](https://github.com/nicolay-r/arekit-ss/blob/master/arekit_ss/sources/src_list.py).
    * `terms_per_context` -- amount of words (terms) in between SOURCE and TARGET objects.
    * `object-source-types` -- filter specific source object types
    * `object-target-types` -- filter specific target object types
    * `relation_types` -- list of types, in which items separated with `|` char; all by default.
    * `splits` -- Manual selection of the data-types related splits that should be chosen for the sampling process; types should be separated by ':' sign; for example: 'train:test'
* `sampler` -- List of the supported samplers:
    * `nn` -- CNN/LSTM architecture related, including frames annotation from [RuSentiFrames](https://github.com/nicolay-r/RuSentiFrames).
        * `no-vectorize` -- flag is applicable only for `nn`, and denotes no need to generate embeddings for features
    * `bert` -- BERT-based, single-input sequence.
    * `prompt` -- prompt-based sampler for LLM systems [[prompt engeneering guide]](https://github.com/dair-ai/Prompt-Engineering-Guide)
        * `prompt` -- For the `prompt` sampler, text of the prompt.
* `writer` -- the output format of samples:
    * `csv` -- for [AREnets](https://github.com/nicolay-r/AREnets) framework;
    * `jsonl` -- for [OpenNRE](https://github.com/thunlp/OpenNRE) framework.
    * `sqlite` -- SQLite-3.0 database.
* `mask_entities` -- mask entity mode.
* Text translation parameters:
    * `src_lang` -- original language of the text.
    * `dest_lang` -- target language of the text.
* `output_dir` -- target directory for samples storing
* Limiting the amount of documents from source:
    * `docs_limit` -- amount of documents to be considered for sampling from the whole source.
    * `doc_ids` -- list of the document IDs.
</details>

![output_prompts](https://github.com/nicolay-r/arekit-ss/assets/14871187/d1499f24-b2df-410b-98cc-8d4018de8c65)

## Powered by

* [AREkit framework](https://github.com/nicolay-r/AREkit)
