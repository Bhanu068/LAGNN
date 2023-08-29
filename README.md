# Language Independent Neuro-Symbolic Semantic Parsing for Form Understanding
This repo is the official implementation of the paper:
Bhanu Prakash Voutharoja, Lizhen Qu, and Fatemeh Shiri. [Language Independent Neuro-Symbolic Semantic Parsing for Form Understanding](https://arxiv.org/pdf/2305.04460.pdf). ICDAR 2023

## Introduction
<p align="center">
    <img src="images/lagnn.png" width = "600"/>
</p>

> Recent works on form understanding mostly employ multimodal transformers or large-scale pre-trained language models. These models need ample data for pre-training. In contrast, humans can usually identify key-value pairings from a form only by looking at layouts, even if they don't comprehend the language used. No prior research has been conducted to investigate how helpful layout information alone is for form understanding. Hence, we propose a unique entity-relation graph parsing method for scanned forms called LAGNN, a language-independent Graph Neural Network model. Our model parses a form into a word-relation graph in order to identify entities and relations jointly and reduce the time complexity of inference. This graph is then transformed by deterministic rules into a fully connected entity-relation graph. Our model simply takes into account relative spacing between bounding boxes from layout information to facilitate easy transfer across languages. To further improve the performance of LAGNN, and achieve isomorphism between entity-relation graphs and word-relation graphs, we use integer linear programming (ILP) based inference.

## Installation
Download all the python packages and dependencies by running this cmd:
```
bash setup.sh
```

## Data Preparation

Download all the data files from [drive.](https://drive.google.com/drive/folders/1BXqWqCg1a6AuxQpeGL6ZjeZ9MCqOQ0r-?usp=sharing)
Alternatively, the files under form_graphs folder are automatically generated during the training phase.

## Run

Run the training with this cmd:
```
bash run.sh
```

Run the testing with this cmd:
```
bash test.sh
```

Run the ILP inference with this cmd:
```
bash run_ilp.sh
```
## Pending Work

- [ ] Code for constraint 1
- [ ] Code to reproduce Table 2

## Citation
If you find this work or code is helpful in your research, please cite:
```bibtex
@InProceedings{lagnn,
author="Voutharoja, Bhanu Prakash
and Qu, Lizhen
and Shiri, Fatemeh",
title="Language Independent Neuro-Symbolic Semantic Parsing for Form Understanding",
booktitle="Document Analysis and Recognition - ICDAR 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="130--146",
isbn="978-3-031-41679-8"
}
```
