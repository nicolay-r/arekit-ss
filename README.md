## arekit-ss 0.23.1

![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.23.1-orange.svg)
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/arekit-googletrans-sampler/blob/master/arekit_googletrans_sampler.ipynb)

> **Mind the case (issue [#18](https://github.com/nicolay-r/arekit-googletrans-sampler/issues/18)):** switching to another language may changed amount of extracted data due to `terms_per_context` parameter
that crops context with fixed amount of words.
> 
<p align="center">
    <img src="logo.png"/>
</p>

arekit-ss [AREkit double "s"] -- extension for instant object-pair context sampling from 
[AREkit](https://github.com/nicolay-r/AREkit)
collection of 
[datasources](https://github.com/nicolay-r/AREkit/wiki/Binded-Sources).

For custom text sampling, please follow the 
[ARElight](https://github.com/nicolay-r/ARElight)
project.

## Installation

Install dependencies:
```bash
pip install git+https://github.com/nicolay-r/arekit_ss.git@master
```

Download AREkit related data, from which `sources` are required:
```bash
python -m arekit.download_data
```

## Usage
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/arekit-googletrans-sampler/blob/master/arekit_googletrans_sampler.ipynb)

```bash
python3 -m arekit_ss.sample --writer csv --source rusentrel --sampler bert \
                            --dest_lang en --docs_limit 1 --text_parser lm --output_dir 'out' 
```
In this example, we convert the RuSentRel collection, originally written in Russian, to the English version,
sampled for the BERT-based models, with samples limited by `limit` parameter

**samplers**:
* `nn` -- CNN/LSTM architecture related, including frames annotation from [RuAttitudes](https://github.com/nicolay-r/RuAttitudes).
* `bert` -- BERT-based, single-input sequence.  
* `prompt` -- prompt-based sampler for `ChatGPT` and the related conversational systems [[prompt engeneering guide]](https://github.com/dair-ai/Prompt-Engineering-Guide)

**writer**:
* `csv` -- for AREkit/AREnets framework;
* `jsonl` -- for [OpenNRE](https://github.com/thunlp/OpenNRE) framework.

## Powered by

* [AREkit framework](https://github.com/nicolay-r/AREkit)
