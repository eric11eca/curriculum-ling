"""Semantic Concept-Relation-Graph (1) Edge Probing task.
Task source paper: https://arxiv.org/pdf/1905.06316.pdf.
Task data prep directions: https://github.com/nyu-mll/jiant/blob/master/probing/data/README.md.
"""
from dataclasses import dataclass

from jiant.tasks.lib.templates.shared import labels_to_bimap
from jiant.tasks.lib.templates import edge_probing_single_span
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(edge_probing_single_span.Example):
    @property
    def task(self):
        return ContradictVertexTask


@dataclass
class TokenizedExample(edge_probing_single_span.TokenizedExample):
    pass


@dataclass
class DataRow(edge_probing_single_span.DataRow):
    pass


@dataclass
class Batch(edge_probing_single_span.Batch):
    pass


class ContradictVertexTask(edge_probing_single_span.AbstractProbingTask):
    Example = Example
    TokenizedExample = TokenizedExample
    DataRow = DataRow
    Batch = Batch

    # LABELS = [
    #    "antonym1",
    #    "antonym2",
    #    "negative1",
    #    "negative2",
    #    "general1",
    #    "general2",
    #    "unalign_word"
    # ]
    LABELS = [
        "contradict1",
        "contradict2",
        "unalign_word"
    ]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    @property
    def num_spans(self):
        return 1

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []

        def reduce_label(label):
            if label in ["antonym1", "negative1", "general1"]:
                return "contradict1"
            elif label in ["antonym2", "negative2", "general2"]:
                return "contradict2"
            else:
                return label

        for (line_num, line) in enumerate(lines):
            # A line in the task's data file can contain multiple targets (span-pair + labels).
            # We create an example for every target:
            for (target_num, target) in enumerate(line["targets"]):
                span = [int(target["span"][0]), int(target["span"][1])]
                examples.append(
                    Example(
                        guid="%s-%s-%s" % (set_type, line_num, target_num),
                        text=line["text"],
                        span=span,
                        labels=[reduce_label(target["label"])] if set_type != "test" else [
                            cls.LABELS[-1]],
                    )
                )
        return examples
