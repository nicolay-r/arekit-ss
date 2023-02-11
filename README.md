## Sentiment Attitude Extraction Resources Translation 

![](https://img.shields.io/badge/Python-3.6-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.23.0-orange.svg)
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/arekit-googletrans-sampler/blob/master/arekit_googletrans_sampler.ipynb)

<p align="center">
    <img src="logo.png"/>
</p>

> **Mind the case (issue [#18](https://github.com/nicolay-r/arekit-googletrans-sampler/issues/18)):** switching to another language may changed amount of extracted data due to `terms_per_context` parameter
that crops context with fixed amount of words.

As a task, [*Sentiment Attitude Extraction*](http://nlpprogress.com/russian/sentiment-analysis.html) 
is devoted to extraction of the sentiment connections from 
subjects towards objects mentioned in texts, usually analytical articles.
This task has been originally proposed and becomes a part of the studies in 
[RuSentRel](https://paperswithcode.com/dataset/rusentrel)
dataset, in which
texts are written in Russian.

To address this limitation, this repository provides `googletrans`-based transfer for 
the result samples that might be composed for any other language you want!

## Installation

Install dependencies:
```bash
pip install -r dependencies.txt
```

Download AREkit related data, from which `sources` are required:
```python
python -m arekit.download_data
```

## Usage
[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/arekit-googletrans-sampler/blob/master/arekit_googletrans_sampler.ipynb)

```bash
python3 translate.py --writer csv --source rusentrel --sampler bert \
                     --dest_lang en --docs_limit 1 --output_dir 'out'
```
In this example, we convert the RuSentRel collection, originally written in Russian, to the English version,
sampled for the BERT-based models, with samples limited by `limit` parameter

**Supported sources**: 
* [RuSentRel](https://paperswithcode.com/dataset/rusentrel)
* [RuAttitudes](https://github.com/nicolay-r/RuAttitudes)
* SentiNEREL

**samplers**:
* nn -- cnn/lstm architecture related, including frames annotation from RuAttitudes.
* bert -- BERT-based, single-input sequence.  

**writer**:
* `csv` -- for AREkit/AREnets framework;
* `jsonl` -- for [OpenNRE](https://github.com/thunlp/OpenNRE) framework.

## References

* [AREkit framework](https://github.com/nicolay-r/AREkit)
