import os
import re
import nltk
import torch
import argparse

import numpy as np
import allennlp_models.tagging
import jiant.utils.python.io as py_io

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rich.progress import track
from allennlp.predictors.predictor import Predictor

stop_words = set(stopwords.words('english'))

MODEL_NAMES = {
    "bert1": "bert-base-uncased",
    "bert2": "bert-large-uncased",
    "roberta1": "roberta-base",
    "roberta2": "roberta-large",
    "deberta": "microsoft/deberta-base",
}

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")


def get_lexical_density(sentence):
    word_tokens = word_tokenize(sentence)
    num_content_words = 0
    for w in word_tokens:
        if not w.lower() in stop_words:
            num_content_words += 1
    return round(num_content_words / len(word_tokens)) * 100


def get_type_token_ratio(sentence):
    sentence = re.sub(r'[^\w]', ' ', sentence).lower()
    tokens = nltk.word_tokenize(sentence)
    types = nltk.Counter(tokens)
    return round(len(types)/len(tokens), 3) * 100


def count_freq(pat, txt):
    M = len(pat)
    N = len(txt)
    res = 0
    for i in range(N - M + 1):
        j = 0
        while j < M:
            if (txt[i + j] != pat[j]):
                break
            j += 1
        if (j == M):
            res += 1
            j = 0
    return res


def count_semantic_roles(sentence):
    sem_roles = predictor.predict(sentence=sentence)
    role_count = 0
    for verb in sem_roles['verbs']:
        role_count += count_freq('ARG', verb['description']) + 1
    return role_count


def compute_metrics_batch(examples):
    metrics = {}
    for i, example in track(enumerate(examples), description="Computing...", total=len(examples)):
        sentence = example['text']
        lex_density = get_lexical_density(sentence)
        lex_diversity = get_type_token_ratio(sentence)
        num_sem_role = count_semantic_roles(sentence)
        metrics[i] = {'text': sentence,
                      'lex_density': lex_density,
                      'num_sem_role': num_sem_role,
                      'lex_diversity': lex_diversity}
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="semgraph2",
                        help="probing task name")
    parser.add_argument("--model_name", type=str, default="bert1",
                        help="pre-trained transformer model name")

    args = parser.parse_args()
    task_name = args.task_name
    model_name = MODEL_NAMES[args.model_name]

    examples = py_io.read_jsonl(f'/content/tasks/data/{task_name}/val.jsonl')
    metrics = compute_metrics_batch(examples)
    os.makedirs(f'./analyze/{task_name}/', exist_ok=True)
    py_io.write_json(metrics, f'./analyze/{task_name}/val_metric.json')
