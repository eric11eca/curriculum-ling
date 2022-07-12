import numpy as np
import os

import jiant.utils.python.io as py_io
import jiant.proj.main.export_model as export_model


MODEL_NAMES = {
    "roberta-mnli": "roberta-large",
    "roberta-anli": "roberta-large",
    "roberta-anli-mix": "roberta-large",
    "deberta": "microsoft/deberta-base",
    "debertav3": "microsoft/deberta-v3-base",
}

HF_MODEL_PATH = {
    "roberta-mnli": "roberta-large-mnli",
    "roberta-anli": "adversarial_nli_r3/roberta-large/main/",
    "roberta-anli-mix": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "deberta": "microsoft/deberta-base",
    "debertav3": "microsoft/deberta-v3-base",
}

MODEL_VAL_NAMES = {
    "roberta-mnli": "roberta-mnli",
    "roberta-anli": "roberta-anli",
    "roberta-anli-mix": "anli-mix-roberta",
    "deberta": "deberta-base",
    "debertav3": "deberta-v3-base",
}


class ModelNotExsitError(Exception):
    """Exception raised for key errors with model names.

    Attributes:
        model_name -- model name causing exception
        message -- explanation of the error
    """

    def __init__(self, model_name, message="does not exist"):
        self.mode_name = model_name
        self.message = f"{model_name} {message}"
        super().__init__(self.message)


def setup_model(model_name, hf_model_name):
    export_model.export_model(
        hf_pretrained_model_name_or_path=hf_model_name,
        output_base_path=f"./models/{model_name}",
    )


def get_k_shot_data_multi(train_lines, seed=42, k=10):
    np.random.seed(seed)
    np.random.shuffle(train_lines)
    label_list = {}
    for line in train_lines:
        label = line['gold_label']
        if label not in label_list:
            label_list[label] = [line]
        else:
            label_list[label].append(line)
    new_train = []
    for label in label_list:
        for line in label_list[label][:k]:
            new_train.append(line)
    return new_train


def task_combinator(tasks):
    train_data_collection = []
    val_data_collection = []

    for task in tasks:
        train_data = py_io.read_jsonl(
            f"/content/tasks/curriculum/{task}/train.jsonl")
        val_data = py_io.read_jsonl(
            f"/content/tasks/curriculum/{task}/val.jsonl")
        train_data_collection += train_data
        val_data_collection += val_data_collection

    task_name = "curriculum"
    os.makedirs("/content/tasks/configs/", exist_ok=True)
    os.makedirs(f"/content/tasks/curriculum/{task_name}", exist_ok=True)
    py_io.write_jsonl(
        data=train_data,
        path=f"/content/tasks/curriculum/{task_name}/train.jsonl",
    )
    py_io.write_jsonl(
        data=val_data,
        path=f"/content/tasks/curriculum/{task_name}/val.jsonl",
    )
    py_io.write_json({
        "task": f"{task_name}",
        "paths": {
            "train": f"/content/tasks/curriculum/{task_name}/train.jsonl",
            "val": f"/content/tasks/curriculum/{task_name}/val.jsonl",
        },
        "name": f"{task_name}"
    }, f"/content/tasks/configs/{task_name}_config.json")
