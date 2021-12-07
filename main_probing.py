import os
import argparse
import numpy as np

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.runscript as main_runscript
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.export_model as export_model
import jiant.utils.display as display
import jiant.utils.python.io as py_io

lexical_tasks = [
    "lexical",
    "transitive",
    "hypernymy",
    "hyponymy",
    "ner"
]

syntactic_tasks = [
    "verbnet",
    "verbcorner",
    "syntactic_alternation",
    "syntactic_variation",
]

logical_tasks = [
    "boolean",
    "comparative",
    "conditional",
    "counting",
    "negation",
    "quantifier",
    "monotonicity_infer",
    "syllogism"
]

semantic_tasks = [
    "sentiment",
    "kg_relations",
    "puns",
    "coreference",
    "context_align",
    "sprl"
]

knowledge_tasks = [
    "entailment_tree"
]

commonsense_tasks = [
    "socialqa",
    "physicalqa",
    "atomic",
    "social_chem"
]

comprehension_tasks = [
    "logiqa",
    "cosmoqa",
    "ester"
]

ANLI = ['adversarial_nli_r1',
        'adversarial_nli_r2',
        'adversarial_nli_r3']

CURRICULUM = lexical_tasks + syntactic_tasks + logical_tasks + knowledge_tasks \
    + semantic_tasks + commonsense_tasks + comprehension_tasks

val_task_dict = {
    "lexical": lexical_tasks,
    "syntactic": syntactic_tasks,
    "logic": logical_tasks,
    "semantic": semantic_tasks,
    "knowledge": knowledge_tasks,
    "commonsense": commonsense_tasks,
    "comprehension": comprehension_tasks
}


task_model_dict = {
    "lexical": "mnli",
    "transitive": "mnli",
    "hypernymy": "mnli",
    "hyponymy": "mnli",
    "ner": "mnli",
    "verbnet": "mnli",
    "verbcorner": "mnli",
    "syntactic_alternation": "mrpc",
    "syntactic_variation": "mrpc",
    "boolean": "mnli",
    "comparative": "mnli",
    "conditional": "mnli",
    "counting": "mnli",
    "negation": "mnli",
    "quantifier": "mnli",
    "monotonicity_infer": "mnli",
    "syllogism": "mnli",
    "sentiment": "mnli",
    "kg_relations": "mnli",
    "puns": "mnli",
    "coreference": "wnli",
    "context_align": "mnli",
    "sprl": "mnli",
    "entailment_tree": "mnli",
    "socialqa": "mnli",
    "physicalqa": "mnli",
    "atomic": "mnli",
    "social_chem": "mnli",
    "logiqa": "mnli",
    "cosmoqa": "mnli",
    "ester": "mnli"
}


def load_data_from_json(train_pth, val_pth):
    train_data = []
    val_data = []
    if len(train_pth) > 0:
        relation_data = py_io.read_json(train_pth, mode="r")
        accum = 0
        for key in relation_data.keys():
            train_data.append(relation_data[key])
            accum += 1
            if accum > 10000:
                break
    if len(val_pth) > 0:
        relation_data_val = py_io.read_json(val_pth, mode="r")
        for key in relation_data_val.keys():
            val_data.append(relation_data_val[key])
    return train_data, val_data


def write_data_to_task_dir(train_data, val_data):
    os.makedirs("/content/tasks/configs/", exist_ok=True)
    os.makedirs("/content/tasks/data/semgraph2", exist_ok=True)
    if len(train_data) > 0:
        py_io.write_jsonl(
            data=train_data,
            path="/content/tasks/data/semgraph2/train.jsonl",
        )
    if len(val_data) > 0:
        py_io.write_jsonl(
            data=val_data,
            path="/content/tasks/data/semgraph2/val.jsonl",
        )
    py_io.write_json({
        "task": "semgraph2",
        "paths": {
            "train": "/content/tasks/data/semgraph2/train.jsonl",
            "val": "/content/tasks/data/semgraph2/val.jsonl",
        },
        "name": "semgraph2"
    }, "/content/tasks/configs/semgraph2_config.json")


def prepare_train_and_val_data(train_pth, val_pth):
    train, val = load_data_from_json(train_pth, val_pth)
    write_data_to_task_dir(train, val)


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


def tokenization(task_name, model_name, control=False, phase=["val"], k_shot=0):
    output_dir = f"./cache/{model_name}/{task_name}"
    if control:
        output_dir = f"./cache/control/{model_name}/{task_name}"
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"/content/tasks/configs/{task_name}_config.json",
        hf_pretrained_model_name_or_path=model_name,
        output_dir=output_dir,
        phases=phase,
        k_shot=k_shot
    ))


def train_configuration(
        task_name, model_name,
        cache_pth, val_task_key,
        classifier_type="linear", do_control=False):
    if do_control:
        task_cache_base_path = f"./cache/control/{model_name}/"
    else:
        task_cache_base_path = cache_pth

    """val_tasks = [task_name]
    if val_task_key != "NONE":
        val_tasks += val_task_dict[val_task_key]"""

    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="/content/tasks/configs/",
        task_cache_base_path=task_cache_base_path,
        train_task_name_list=[task_name],
        val_task_name_list=[task_name],
        train_batch_size=8,
        eval_batch_size=32,
        epochs=5,
        num_gpus=1,
        classifier_type=classifier_type
    ).create_config()

    for task in jiant_run_config["taskmodels_config"]["task_to_taskmodel_map"]:
        jiant_run_config["taskmodels_config"]["task_to_taskmodel_map"][task] = task_name
    # jiant_run_config["taskmodels_config"]["task_to_taskmodel_map"] = task_model_dict

    os.makedirs("./run_configs/", exist_ok=True)
    py_io.write_json(
        jiant_run_config,
        f"./run_configs/{task_name}_run_config.json")
    display.show_json(jiant_run_config)


def train(
    task_name, model_name,
    model_path, model_dir_name,
    model_load_mode="from_transformers",
    do_train=True, freeze_encoder=False
):

    output_dir = f"./runs/{task_name}/{model_dir_name}/main"
    os.makedirs(output_dir, exist_ok=True)
    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=f"./run_configs/{task_name}_run_config.json",
        output_dir=output_dir,
        hf_pretrained_model_name_or_path=model_name,
        model_load_mode=model_load_mode,
        model_path=model_path,
        model_config_path=f"./models/{model_name}/model/config.json",
        learning_rate=1e-5,
        eval_every_steps=1000,
        do_train=do_train,
        do_val=True,
        do_save_best=True,
        write_val_preds=True,
        freeze_encoder=freeze_encoder,
        force_overwrite=True,
        no_cuda=False
    )
    main_runscript.run_loop(run_args)


def setup_model(model_name):
    export_model.export_model(
        hf_pretrained_model_name_or_path=model_name,
        output_base_path=f"./models/{model_name}",
    )


MODEL_NAMES = {
    "bert1": "bert-base-uncased",
    "bert2": "bert-large-uncased",
    "roberta1": "roberta-base",
    "roberta1-glue": "roberta-base",
    "roberta2": "roberta-large",
    "roberta-anli": "anli-mix-roberta",
    "deberta": "microsoft/deberta-base",
    "debertav3": "microsoft/deberta-v3-base",
}

MODEL_VAL_NAMES = {
    "bert1": "bert-base",
    "bert2": "bert-large",
    "roberta1": "roberta-base",
    "roberta2": "roberta-large",
    "roberta-anli": "anli-mix-roberta",
    "roberta1-glue": "roberta-base",
    "deberta": "deberta-base",
    "debertav3": "deberta-v3-base",
    "roberta-glue": "roberta-glue",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pth", type=str, default="",
                        help="path to training data json file")
    parser.add_argument("--val_pth", type=str, default="",
                        help="path to validation data json file")
    parser.add_argument("--tokenize_train", action="store_true",
                        help="enable tokenization and caching of train data")
    parser.add_argument("--tokenize_val", action="store_true",
                        help="enable tokenization and caching of val data")
    parser.add_argument("--tokenize", action="store_true",
                        help="enable tokenization and caching of both train and val data")
    parser.add_argument("--tokenize_control", action="store_true",
                        help="enable tokenization and caching of control val data")
    parser.add_argument("--setup_model", action="store_true",
                        help="enable pre-trained models to setup")
    parser.add_argument("--main_loop", action="store_true",
                        help="enable train-eval runner"),
    parser.add_argument("--load_best", action="store_true",
                        help="load the best checkpoint instead of base model"),
    parser.add_argument("--task_name", type=str, default="semgraph2",
                        help="probing task name")
    parser.add_argument("--k_shot", type=int, default=0,
                        help="number of data shots")
    parser.add_argument("--exp_list", action='append',
                        help="probing experiments name")
    parser.add_argument("--model_name", type=str, default="bert1",
                        help="pre-trained transformer model name")

    args = parser.parse_args()
    task_name = args.task_name
    exp_name = args.model_name
    model_name = MODEL_NAMES[args.model_name] if args.model_name in MODEL_NAMES else ""
    print("Task Name: ", task_name)
    print("Model Name: ", model_name)

    if len(args.train_pth) > 0 or len(args.val_pth) > 0:
        prepare_train_and_val_data(args.train_pth, args.val_pth)

    if args.tokenize:
        if task_name == "curriculum":
            for task in CURRICULUM:
                tokenization(
                    task, model_name, phase="train",
                    k_shot=args.k_shot)
                tokenization(task, model_name, phase="val", k_shot=args.k_shot)
        else:
            tokenization(
                task_name, model_name,
                phase="train", k_shot=args.k_shot)
            tokenization(
                task_name, model_name,
                phase="val", k_shot=args.k_shot)
    elif args.tokenize_train:
        if task_name == "curriculum":
            for task in CURRICULUM:
                tokenization(
                    task, model_name, phase="train",
                    k_shot=args.k_shot)
                #tokenization(task, model_name, phase="val", k_shot=args.k_shot)
        else:
            tokenization(
                task_name, model_name,
                phase="train", k_shot=args.k_shot)
            #tokenization(task_name, model_name, phase="val", k_shot=args.k_shot)
    elif args.tokenize_val:
        if task_name == "curriculum":
            for task in CURRICULUM:
                #tokenization(task, model_name, phase="train", k_shot=args.k_shot)
                tokenization(task, model_name, phase="val", k_shot=args.k_shot)
        else:
            #tokenization(task_name, model_name, phase="train", k_shot=args.k_shot)
            tokenization(
                task_name, model_name,
                phase="val", k_shot=args.k_shot)
    elif args.tokenize_control:
        tokenization(
            task_name, model_name, control=True,
            phase="val", k_shot=args.k_shot)

    if args.setup_model:
        setup_model(model_name)

    if args.main_loop:

        model_val_name = MODEL_VAL_NAMES[args.model_name]
        model_path = f"./models/{model_name}/model/model.p"
        #model_path = f"./runs/fundamental/{model_val_name}/main/best_model.p"
        if args.load_best:
            model_path = f"./runs/{task_name}/{model_val_name}/main/best_model.p"
        cache_path = f"./cache/{model_name}/"

        if "inference" in task_name:
            val_task_key = task_name.replace("_inference", "")
        else:
            val_task_key = "NONE"

        load_mode = "from_transformers"
        if args.load_best:
            load_mode = "all"

        train_configuration(
            task_name,
            model_name,
            cache_pth=cache_path,
            val_task_key=val_task_key,
            classifier_type="mlp",
            do_control=False
        )

        train(
            task_name=task_name,
            model_name=model_name,
            model_path=model_path,
            model_dir_name=model_val_name,
            model_load_mode=load_mode,
            do_train=True,
            freeze_encoder=True,
        )

# python main_probing.py --main_loop --task_name monotonicity --exp_list bert2-mlp --exp_list roberta2 --exp_list roberta2-mlp --exp_list bert2
