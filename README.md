# Curriculum: A Broad-Coverage Benchmark for Linguistic Phenomena in Natural Language Understanding

Repository for the paper [Curriculum: A Broad-Coverage Benchmark for Linguistic Phenomena in Natural Language Understanding](https://aclanthology.org/2022.naacl-main.234), in Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies ([NAACL-HLT 2022](https://aclanthology.org/volumes/2022.naacl-main/)).

TLDR; Curriculum is a new format of NLI benchmark for evaluation of broad-coverage linguistic phenomena. This linguistic-phenomena-driven benchmark can serve as an effective tool for diagnosing model behavior and verifying model learning quality.

## Background
### Why linguistic-phemomena-driven benchmark?
In the age of large transformer language models, linguistic evaluation play an important role in diagnosing models’ abilities and limitations on natural language understanding. However, current evaluation methods show some significant shortcomings. In particular, they do not provide insight into how well a language model captures distinct linguistic skills essential for language understanding and reasoning. Thus they fail to effectively map out the aspects of language understanding that remain challenging to existing models, which makes it hard to discover potential limitations in models and datasets. 

### What we intend to contribute?
We desire to initiate Curriculum as a new format of NLI benchmark for evaluation of broad-coverage linguistic phenomena. We hope that a benchmark can be more than just a leaderboard but rather provide insight into the limitation of existing benchmark datasets and state-of-the-art models that may encourage future research on re-designing datasets, model architectures, and learning objectives.

## Introduction
Curriculum contains a collection of datasets that covers 36 types of major linguistic phenomena and an evaluation procedure for diagnosing how well a language model captures reasoning skills for distinct types of linguistic phenomena. We show that this linguistic-phenomena-driven benchmark can serve as an effective tool for diagnosing model behavior and verifying model learning quality. 

## Linguistic Phenomena
![phenomena_schema](https://hackmd.io/_uploads/SyDNCKeH3.png)

## Dataset Statistics
![dataset_stats](https://hackmd.io/_uploads/S1IgXqlr3.png)


## Download Benchmark
You can download the benchmark either from the `benchmark` folder in this repostiroy or through HuggingFace Datasets: [curriculum_benchmark](https://huggingface.co/datasets/chenz16/curriculum_benchmark).


## Referece
Cite this dataset and the paper with:
```bib
@inproceedings{chen-gao-2022-curriculum,
    title = "Curriculum: A Broad-Coverage Benchmark for Linguistic Phenomena in Natural Language Understanding",
    author = "Chen, Zeming  and Gao, Qiyue",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.234",
    doi = "10.18653/v1/2022.naacl-main.234",
    pages = "3204--3219",
}
```
