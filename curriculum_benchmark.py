# Lint as: python3
"""CURRICULUM Benchmark"""

import json
import os

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2204.06283,
  doi = {10.48550/ARXIV.2204.06283},
  url = {https://arxiv.org/abs/2204.06283},
  author = {Chen, Zeming and Gao, Qiyue},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Curriculum: A Broad-Coverage Benchmark for Linguistic Phenomena in Natural Language Understanding},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

_DESCRIPTION = """\
We introduce Curriculum as a new format of NLI benchmark for evaluation of broad-coverage linguistic phenomena. 
Curriculum contains a collection of datasets that covers 36 types of major linguistic phenomena and an evaluation procedure 
for diagnosing how well a language model captures reasoning skills for distinct types of linguistic phenomena. 
We show that this linguistic-phenomena-driven benchmark can serve as an effective tool for diagnosing 
model behavior and verifying model learning quality.
"""

_HOMEPAGE = "https://github.com/eric11eca/curriculum-ling"
_LICENSE = "CC BY-SA 3.0"
_URL = "https://github.com/eric11eca/curriculum-ling/blob/main/benchmark/tasks/"


_DESCRIPTION_MAP = {
    "analytic": "analytical thinking.",
    "atomic": "reasoning on commonsense knowledge graph.",
}

_TAKS_NAMES = ["analytic", "defeasible", "boolean", "comparative",
               "conditional", "context_align", "control", "coreference",
               "cosmoqa", "counterfactual", "counting", "drop",
               "entailment_tree", "ester", "hellaswag", "hypernymy",
               "hyponymy", "kg_relations", "lexical", "logiqa",
               "monotonicity_infer", "negation", "ner", "physicalqa",
               "puns", "quantifier", "sentiment", "socialqa",
               "spatial", "sprl", "syntactic_alternation", "syntactic_variation",
               "temporal", "transitive", "verbcorner", "verbnet"]

task_label_dict = {
    "lexical": ["entailed", "not-entailed"],
    "transitive": ["entailed", "not-entailed"],
    "hypernymy": ["entailed", "not-entailed"],
    "hyponymy": ["entailed", "not-entailed"],
    "ner": ["entailed", "not-entailed"],
    "verbnet": ["entailed", "not-entailed"],
    "verbcorner": ["entailed", "not-entailed"],
    "syntactic_alternation": ["entailed", "not-entailed"],
    "syntactic_variation": ["entailed", "not-entailed"],
    "boolean": ["entailment", "contradiction", "neutral"],
    "comparative": ["entailment", "contradiction", "neutral"],
    "conditional": ["entailment", "contradiction", "neutral"],
    "counting": ["entailment", "contradiction", "neutral"],
    "negation": ["entailment", "contradiction", "neutral"],
    "quantifier": ["entailment", "contradiction", "neutral"],
    "monotonicity_infer": ["entailed", "not-entailed"],
    "sentiment": ["entailed", "not-entailed"],
    "kg_relations": ["entailed", "not-entailed"],
    "puns": ["entailed", "not-entailed"],
    "coreference": ["entailed", "not-entailed"],
    "context_align": ["entailed", "not-entailed"],
    "sprl": ["entailed", "not-entailed"],
    "analytic": ["entailed", "not-entailed"],
    "entailment_tree": ["entailed", "not-entailed"],
    "socialqa": ["entailed", "not-entailed"],
    "physicalqa": ["entailed", "not-entailed"],
    "hellaswag": ["entailed", "not-entailed"],
    "cosmoqa": ["entailed", "not-entailed"],
    "logiqa": ["entailed", "not-entailed"],
    "ester": ["entailed", "not-entailed"],
    "drop": ["entailed", "not-entailed"],
    "control": ["entailment", "contradiction", "neutral"],
    "spatial": ["entailed", "not-entailed"],
    "temporal": ["entailed", "not-entailed"],
    "defeasible": ["entailed", "not-entailed"],
    "counterfactual": ["entailed", "not-entailed"]
}


def read_file(path, mode="r", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        return f.read()


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def read_json(path, mode="r", **kwargs):
    return json.loads(read_file(path, mode=mode, **kwargs))


def write_json(data, path):
    return write_file(json.dumps(data, indent=2), path)


def read_jsonl(path, mode="r", **kwargs):
    # Manually open because .splitlines is different from iterating over lines
    ls = []
    with open(path, mode, **kwargs) as f:
        for line in f:
            ls.append(json.loads(line))
    return ls


def write_jsonl(data, path):
    assert isinstance(data, list)
    lines = [to_jsonl(elem) for elem in data]
    write_file("\n".join(lines), path)


def to_jsonl(data):
    return json.dumps(data).replace("\n", "")


class CurriculumConfig(datasets.BuilderConfig):
    """BuilderConfig for Curriculum."""

    def __init__(self, features, data_url, citation, url, label_classes=["entailed", "not-entailed"], **kwargs):
        """BuilderConfig for Curriculum.
        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
          data_url: `string`, url to download the zip file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          label_classes: `list[string]`, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
          **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 1.0.0: Initial version.
        super(CurriculumConfig, self).__init__(
            version=datasets.Version("1.0.0"), **kwargs)
        self.features = features
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url


class CurriculumBenchmark(datasets.GeneratorBasedBuilder):
    """Curriculum Benchmark. Version 1.0.0"""

    BUILDER_CONFIGS = [
        CurriculumConfig(
            name=task_name,
            description=_DESCRIPTION,
            label_classes=task_label_dict[task_name],
            features=["premise", "hypothesis", "idx", "gold_label"],
            data_url=f"https://github.com/eric11eca/curriculum-ling/raw/main/benchmark/tasks/{task_name}.zip",
            citation=_CITATION,
            url="https://github.com/eric11eca/curriculum-ling/",
        ) for task_name in _TAKS_NAMES
    ]

    def _info(self):
        features = {feature: datasets.Value(
            "string") for feature in self.config.features}
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    @staticmethod
    def _get_filepath(dl_dir, split):
        return os.path.join(dl_dir, split + ".jsonl")

    def _split_generators(self, dl_manager):
        dl_dir = dl_manager.download_and_extract(self.config.data_url) or ""
        task_name = _get_task_name_from_data_url(self.config.data_url)
        dl_dir = os.path.join(dl_dir, task_name)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "train.jsonl"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "val.jsonl"),
                    "split": datasets.Split.VALIDATION,
                },
            )
        ]

    def _generate_examples(self, data_file, split):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_file)

        dataset = read_jsonl(data_file)
        for id_, data in enumerate(dataset):

            yield id_, {
                "premise": data["premise"],
                "hypothesis": data["hypothesis"],
                "gold_label": data["gold_label"],
                "idx": id_
            }


def _get_task_name_from_data_url(data_url):
    return data_url.split("/")[-1].split(".")[0]
