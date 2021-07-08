import argparse
import os
import time
import torch
from utils import utils
from dataset import load_data
from extract import extract
from evaluate import Benchmark
from evaluate import Matcher
from evaluate import GeneralReader


def get_performance(output_path, gold_path):
    auc, precision, recall, f1 = [None for _ in range(4)]
    matching_func = Matcher.lexicalMatch
    error_fn = os.path.join(output_path, 'error_idxs.txt')

    evaluator = Benchmark(gold_path)
    reader = GeneralReader()
    reader.read(os.path.join(output_path, 'extraction.txt'))

    (precision, recall, f1), auc = evaluator.compare(
        predicted=reader.oie,
        matchingFunc=matching_func,
        output_fn=os.path.join(output_path, 'pr_curve.txt'),
        error_file=error_fn)
    return auc, precision, recall, f1


def do_eval(output_path, gold_path):
    auc, prec, rec, f1 = get_performance(output_path, gold_path)
    eval_results = [f1, prec, rec, auc]
    return eval_results
