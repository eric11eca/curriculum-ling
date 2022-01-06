import os
import argparse
import logging
import torch
import numpy as np

from operator import itemgetter
from sklearn.model_selection import train_test_split

import jiant.utils.python.io as py_io
import jiant.utils.python.filesystem as filesystem

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_zlog(fol_path):
    all_paths = filesystem.find_files_with_ext(fol_path, "zlog")
    log_data = {}
    for path in all_paths:
        key = os.path.abspath(path).replace(os.path.abspath(fol_path), "")[
            1:].replace(".zlog", "")
        log_data[key] = py_io.read_jsonl(path)
    return log_data


def load_chunk(fol_path):
    all_paths = filesystem.find_files_with_ext(fol_path, "chunk")
    data_chunks = {}
    for path in all_paths:
        key = os.path.abspath(path).replace(os.path.abspath(fol_path), "")[
            1:].replace(".chunk", "")
        data_chunks[key] = torch.load(path)
    return data_chunks


def count_labels(datalist):
    label_list = {}
    for line in datalist:
        label = line['gold_label']
        if label not in label_list:
            label_list[label] = 1
        else:
            label_list[label] += 1
    print(label_list)


def read_training_dynamics(
        task_name, model_name, phase_name, split_name="training",
        strip_last=False, id_field="guid", burn_out=None):
    """
    Given path to logged training dynamics, merge stats across epochs.
    Returns:
    - Dict between ID of a train instances and its gold label, and the list of logits across epochs.
    """
    train_dynamics = {}

    log_path = f"./runs/{task_name}/{model_name}/{phase_name}/"
    dynamic_zlogs = load_zlog(log_path)
    num_epochs = len(dynamic_zlogs)
    if burn_out:
        num_epochs = burn_out
    logger.info(f"Reading {num_epochs} dynamic logs for {task_name} ...")

    dynamic_log = dynamic_zlogs[f"{split_name}_dynamic"]
    for record in dynamic_log:
        guid = record[id_field] if not strip_last else record[id_field][:-1]
        if guid not in train_dynamics:
            train_dynamics[guid] = {"gold": record["gold"], "logits": []}
        train_dynamics[guid]["logits"].append(record["logits"])

    logger.info(
        f"Read training dynamics for {len(train_dynamics)} instances.")
    return train_dynamics


def compute_pointwise_v_entropy(
    val_labels,
    main_train_dynamics,
    null_train_dynamics
):
    N = len(val_labels)
    val_labels = [[label] for label in val_labels]

    prediction_main = [x['logits'][0] for x in main_train_dynamics.values()]
    prediction_null = [x['logits'][0] for x in null_train_dynamics.values()]
    val_batch_main = torch.Tensor(prediction_main)
    val_batch_null = torch.Tensor(prediction_null)
    label_batch = torch.Tensor(val_labels).type(torch.int64)

    if len(val_batch_main.shape) == 3:
        prediction_batch_main = torch.softmax(val_batch_main, 2)
        prediction_batch_null = torch.softmax(val_batch_null, 2)
        label_batch = label_batch.view(*label_batch.shape, 1)
        prediction_batch_main = torch.gather(
            prediction_batch_main, 2, label_batch)
        prediction_batch_null = torch.gather(
            prediction_batch_null, 2, label_batch)
    else:
        prediction_batch_main = torch.softmax(val_batch_main, 1)
        prediction_batch_null = torch.softmax(val_batch_null, 1)
        prediction_batch_main = torch.gather(
            prediction_batch_main, 1, label_batch)
        prediction_batch_null = torch.gather(
            prediction_batch_null, 1, label_batch)
        label_batch = label_batch.view(label_batch.shape[0])
        label_batch = label_batch.view(*label_batch.shape, 1)

    pvi_main = torch.where(
        (label_batch != 0),
        torch.log2(prediction_batch_main),
        torch.zeros_like(prediction_batch_main)
    )

    pvi_null = torch.where(
        (label_batch != 0),
        torch.log2(prediction_batch_null),
        torch.zeros_like(prediction_batch_null)
    )

    pvi_tensor = -pvi_null + pvi_main
    V_info = (1/N)*torch.sum(pvi_tensor)
    pvi = [x.item() for x in pvi_tensor]
    return V_info, pvi


def dataset_split_by_pvi(pvi_record):
    sorted_by_label = {}
    for label in pvi_record:
        sorted_val_data = sorted(
            pvi_record[label],
            key=itemgetter('pvi'),
            reverse=True)
        pvi = [data['pvi'] for data in sorted_val_data]
        median = min(pvi) + (abs(max(pvi)) + abs(min(pvi))) / 2
        val_data_buckets = {
            "simple": [],
            "hard": []
        }
        if round(max(pvi), 2) == round(min(pvi), 2):
            middle_index = len(sorted_val_data)//2
            val_data_buckets["simple"] = sorted_val_data[:middle_index]
            val_data_buckets["hard"] = sorted_val_data[middle_index:]
        else:
            for data in sorted_val_data:
                if data['pvi'] < median:
                    val_data_buckets["hard"].append(data)
                else:
                    val_data_buckets["simple"].append(data)

        sorted_by_label[label] = val_data_buckets

    simple_split_train = []
    hard_split_train = []
    simple_split_val = []
    hard_split_val = []

    for label in sorted_by_label:
        simple_data = sorted_by_label[label]["simple"]
        print(len(simple_data))
        simple_train, simple_val, _, _ = train_test_split(
            simple_data, [0]*len(simple_data),
            test_size=0.4, random_state=42)
        simple_split_train += simple_train
        simple_split_val += simple_val

        hard_data = sorted_by_label[label]["hard"]
        hard_train, hard_val, _, _ = train_test_split(
            hard_data, [0]*len(hard_data),
            test_size=0.4, random_state=42)
        hard_split_train += hard_train
        hard_split_val += hard_val

    return simple_split_train, hard_split_train, simple_split_val, hard_split_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="boolean",
                        help="curriculum task name")
    parser.add_argument("--task_split", type=str, default="val",
                        help="curriculum task split")
    parser.add_argument("--calc_vinfo", action="store_true",
                        help="calculate dataset and pointwise v_entropy from train/val dynamics"),
    parser.add_argument("--dataset_split", action="store_true",
                        help="split dataset into simple and hard partitions based on pointwise v_entropy"),

    args = parser.parse_args()
    task = args.task_name

    if args.calc_vinfo:
        main_train_dynamics = read_training_dynamics(
            task, "anli-mix-roberta",
            "1000-shot", split_name="val")
        null_train_dynamics = read_training_dynamics(
            task, "anli-mix-roberta",
            "1000-shot-null", split_name="val")

        val_chunks = load_chunk(f"./cache/roberta-large/{task}/val_labels/")
        print(len(val_chunks))
        val_labels = np.concatenate(
            [val_chunks[k] for k in val_chunks], axis=0)

        V_info, pvi = compute_pointwise_v_entropy(
            val_labels, main_train_dynamics, null_train_dynamics)

        logger.info(f"Minimum PVI: {min(pvi)}")
        logger.info(f"Maximum PVI: {max(pvi)}")
        logger.info(f"Dataset Difficulty: {V_info}")

        val_data = py_io.read_jsonl(
            f"/content/tasks/curriculum/{task}/{args.task_split}.jsonl")

        val_data_pvi = []
        for i, data in enumerate(val_data):
            record = data
            record['pvi'] = pvi[i]
            val_data_pvi.append(record)

        py_io.write_jsonl(
            data=val_data_pvi,
            path=f"/content/tasks/curriculum/{task}/{args.task_split}_pvi.jsonl",
        )

        logger.info(
            f"Wrote {task} {args.task_split} pvi, {len(val_data)} instances to file.")

    if args.dataset_split:
        pvi_train = py_io.read_jsonl(
            f"/content/tasks/curriculum/{task}/train_pvi.jsonl")
        pvi_val = py_io.read_jsonl(
            f"/content/tasks/curriculum/{task}/val_pvi.jsonl")
        pvi = pvi_train + pvi_val

        pvi_record = {}
        for i, data in enumerate(pvi):
            label = data['gold_label']
            if label not in pvi_record:
                pvi_record[label] = [data]
            else:
                pvi_record[label].append(data)

        for label in pvi_record:
            print(label)
            pvi = [data['pvi'] for data in pvi_record[label]]
            print(max(pvi))
            print(min(pvi))

        simple_split_train, hard_split_train, simple_split_val, hard_split_val = dataset_split_by_pvi(
            pvi_record)

        print("train split")
        pvi = [data['pvi'] for data in simple_split_train]
        print(max(pvi))
        print(min(pvi))

        pvi = [data['pvi'] for data in hard_split_train]
        print(max(pvi))
        print(min(pvi))

        print("val split")
        pvi = [data['pvi'] for data in simple_split_val]
        print(max(pvi))
        print(min(pvi))

        pvi = [data['pvi'] for data in hard_split_val]
        print(max(pvi))
        print(min(pvi))

        py_io.write_jsonl(
            data=simple_split_train,
            path=f"/content/tasks/curriculum/{task}/train_simple.jsonl",
        )

        py_io.write_jsonl(
            data=hard_split_train,
            path=f"/content/tasks/curriculum/{task}/train_hard.jsonl",
        )

        py_io.write_jsonl(
            data=simple_split_val,
            path=f"/content/tasks/curriculum/{task}/val_simple.jsonl",
        )

        py_io.write_jsonl(
            data=hard_split_val,
            path=f"/content/tasks/curriculum/{task}/val_hard.jsonl",
        )
        logger.info(f"Write dataset splits to file.")
