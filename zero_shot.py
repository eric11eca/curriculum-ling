import json
import argparse
import torch

from tqdm import tqdm
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
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


def count_binary_labels(datalist):
    num_ent = 0
    num_nent = 0

    for data in datalist:
        if data['gold_label'] == "entailed":
            num_ent += 1
        if data['gold_label'] == "not-entailed":
            num_nent += 1

    return max(num_ent, num_nent) / len(datalist)


def count_triple_labels(datalist):
    num_ent = 0
    num_con = 0
    num_neu = 0

    for data in datalist:
        if data['gold_label'] == "entailment":
            num_ent += 1
        if data['gold_label'] == "contradiction":
            num_con += 1
        if data['gold_label'] == "neutral":
            num_neu += 1

    return max(num_ent, num_con, num_neu) / len(datalist)


def model_test(model, tokenizer, test_data, classes, majority=False):
    premises = []
    hypothesis = []
    labels = []
    for data in test_data:
        premises.append(data["premise"])
        hypothesis.append(data["hypothesis"])
        labels.append(classes.index(data["gold_label"]))

    if majority:
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(premises, labels)
        preds = dummy_clf.predict(premises)
        acc = dummy_clf.score(preds, labels)
        mcc = matthews_corrcoef(preds, labels)
    else:
        preds = []
        for i in tqdm(range(len(premises))):  # premises[i],
            test_sentence = tokenizer(
                premises[i], hypothesis[i], max_length=512,
                truncation=True, return_tensors="pt")
            test_sentence.to('cuda')
            logits = model(**test_sentence).logits
            out = torch.softmax(logits, dim=1)
            pred = torch.argmax(out).cpu().numpy()
            if len(classes) == 2:
                pred = min(1, pred)
            preds.append(pred)

        acc = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
    return acc, mcc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="logical",
                        help="zero-shot category name")
    parser.add_argument("--model", type=str, default="logical",
                        help="zero-shot model name")
    parser.add_argument("--majority", action="store_true",
                        help="calculate majority baseline")
    args = parser.parse_args()

    model_dict = {
        "bert_base": "textattack/bert-base-uncased-MNLI",
        "bert_large": "sentence-transformers/bert-large-nli-mean-tokens",
        "roberta_base": "textattack/roberta-base-MNLI",
        "mnli_roberta": "roberta-large-mnli",
        "deberta_base": "microsoft/deberta-base-mnli",
        "mnli_bart": "textattack/facebook-bart-large-MNLI",
        "anli_roberta": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "anli_xlnet": "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli",
        "anli_roberta_small": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    }

    if args.majority:
        model_name = "majority"
        model = None
        tokenizer = None
    else:
        model_name = model_dict[args.model]
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
                "transitive", "hypernymy", "hyponymy", "ner",
                "verbcorner", "verbnet",
                "syntactic_alternation", "syntactic_variation",
                "monotonicity_infer", "syllogism",
                "coreference", "puns", "sentiment",
                "kg_relations", "context_align",  "sprl",
                "atomic", "social_chem", "socialqa", "physicalqa",
                "logiqa", "ester", "entailment_tree", "cosmoqa"
            ],
            "classes": ["entailed", "not-entailed"]
        },
        "binarynli2": {
            "task_names": [
                "monotonicity_infer", "socialqa"
            ],
            "classes": ["entailed", "not-entailed"]
        },
        "high_level": {
            "task_names": [
                "lexical_inference", "syntactic_inference", "logical_inference",
                "semantic_inference", "commonsense_inference"],
            "classes": ["entailed", "not-entailed"]
        }
    }

    task_names = category_map[category]['task_names']
    classes = category_map[category]['classes']
    nli_eval = {}

    for task in task_names:
        print(f"Evaluating on {task} ...")
        test_data = read_jsonl(f"/content/tasks/curriculum/{task}/val.jsonl")

        acc, mcc = model_test(
            model, tokenizer,
            test_data=test_data,
            classes=classes,
            majority=args.majority
        )

        nli_eval[task] = {
            "acc": acc,
            "mcc": mcc
        }

        print(f"Evaluation Result for {task}:")
        print(nli_eval[task])
        write_json(
            nli_eval, f"./runs/zero-shot/{args.model}/{category}_eval.json")

    print(nli_eval)
    write_json(nli_eval, f"./runs/zero-shot/{args.model}/{category}_eval.json")
