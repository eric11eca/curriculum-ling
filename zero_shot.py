import os
import glob
import json
import argparse
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification


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


def to_jsonl(data):
    return json.dumps(data).replace("\n", "")


def write_jsonl(data, path):
    assert isinstance(data, list)
    lines = [to_jsonl(elem) for elem in data]
    write_file("\n".join(lines), path)


def model_test(model, tokenizer, test_data, classes):
    premises = []
    hypothesis = []
    labels = []
    for data in test_data:
        premises.append(data["premise"])
        hypothesis.append(data["hypothesis"])
        labels.append(data["gold_label"])

    num_correct = 0
    for i in tqdm(range(len(premises))):  # premises[i],
        test_sentence = tokenizer(
            premises[i], hypothesis[i], return_tensors="pt")
        test_sentence.to('cuda')
        logits = model(**test_sentence).logits
        out = torch.softmax(logits, dim=1)
        pred = torch.argmax(out).cpu().numpy()
        if len(classes) == 2:
            pred = min(1, pred)
        if not labels[i] in classes:
            print(labels[i])

        if pred == classes.index(labels[i]):
            num_correct += 1

    acc = num_correct * 100 / len(premises)
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="logical",
                        help="zero-shot category name")
    args = parser.parse_args()

    bert_base = "textattack/bert-base-uncased-MNLI"
    #bert_large = "sentence-transformers/bert-large-nli-mean-tokens"
    roberta_base = "textattack/roberta-base-MNLI"
    roberta_large = "roberta-large-mnli"
    deberta_base = "microsoft/deberta-base-mnli"
    #bart_large = "textattack/facebook-bart-large-MNLI"
    #multi_t5 = "bigscience/T0pp"
    anli_roberta = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

    model_name = roberta_large

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.cuda()

    category = args.category

    category_map = {
        "commonnli": {
            "task_names": [
                "lexical",
                "boolean", "comparative", "conditional",
                "counting", "negation", "quantifier",
            ],
            "classes": ["entailment", "neutral", "contradiction"]
        },
        "commonnli2": {
            "task_names": [
                "lexical",
                "boolean", "comparative", "conditional",
                "counting", "negation", "quantifier",
            ],
            "classes": ["entailment", "neutral", "contradiction"]
        },
        "binarynli": {
            "task_names": [
                "transitive", "hypernymy", "hyponymy",
                "verbcorner", "verbnet", "ner",
                "coreference", "puns", "sentiment",
                "monotonicity_infer", "syntactic_alternation",
                "kg_relations", "context_align",  "sprl",
                "atomic", "social_chem", "socialqa",
                "logiqa", "ester", "entailment_tree", "cosmoqa",
                "syllogism"
            ],
            "classes": ["entailed", "not-entailed"]
        },
        "binarynli2": {
            "task_names": [
                "ester", "physicalqa"
            ],
            "classes": ["entailed", "not-entailed"]
        },
        "semantics": {
            "task_names": [
                "coreference", "puns", "sentiment",
                "monotonicity_infer", "syntactic_alternation",
                "kg_relations", "context_align",  "sprl"],
            "classes": ["entailed", "not-entailed"]
        },
        "commonsense": {
            "task_names": [
                "atomic", "social_chem", "socialqa"],
            "classes": ["entailed", "not-entailed"]
        },
        "comprehension": {
            "task_names": [
                "logiqa", "ester", "entailment_tree", "cosmoqa"],
            "classes": ["entailed", "not-entailed"]
        },
    }

    task_names = category_map[category]['task_names']
    classes = category_map[category]['classes']
    nli_eval = {}

    for task in task_names:
        print(f"Evaluating on {task} ...")
        test_data = read_jsonl(f"/content/tasks/curriculum/{task}/val.jsonl")
        acc = model_test(model, tokenizer, test_data, classes)
        nli_eval[task] = acc
        print(f"Evaluation Result for {task}: acc. {acc}")
        write_json(nli_eval, f"./runs/zero-shot/{category}_eval.json")

    print(nli_eval)
    write_json(nli_eval, f"./runs/zero-shot/{category}_eval.json")
