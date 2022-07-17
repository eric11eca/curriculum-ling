import os

import jiant.utils.python.io as py_io

def count_labels(dataset, binary=True):
    if binary:
        label_counter = {
            "entailed": 0,
            "not-entailed": 0
        }
    else:
        label_counter = {
            "entailment": 0,
            "contradiction": 0,
            "neutral": 0
        }

    for data in dataset:
        label_counter[data['gold_label']] += 1

    return label_counter


def write_dataset_to_disk(task_name, train_data, val_data, data_dir="./benchmark/tasks/"):
    os.makedirs(f"{data_dir}/configs/", exist_ok=True)
    os.makedirs(f"{data_dir}/{task_name}", exist_ok=True)
    py_io.write_jsonl(
        data=train_data,
        path=f"{data_dir}/{task_name}/train.jsonl",
    )
    py_io.write_jsonl(
        data=val_data,
        path=f"{data_dir}/{task_name}/val.jsonl",
    )
    py_io.write_json({
        "task": f"{task_name}",
        "paths": {
            "train": f"{data_dir}/{task_name}/train.jsonl",
            "val": f"{data_dir}/{task_name}/val.jsonl",
        },
        "name": f"{task_name}"
    }, f"{data_dir}/configs/{task_name}_config.json")
