import os
import argparse
import numpy as np
import shutil

import jiant.utils.display as display
import jiant.utils.python.io as py_io

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.runscript as main_runscript
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.export_model as export_model

from ray import tune

lexical_tasks = [
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
    "entailment_tree",
    "proof_writer"
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
    "ester",
    "drop"
]

special_tasks = [
    "temporal",
    "spatial",
    "counterfactual"
]

ANLI = ['adversarial_nli_r1',
        'adversarial_nli_r2',
        'adversarial_nli_r3']

CURRICULUM = lexical_tasks + syntactic_tasks + logical_tasks + knowledge_tasks \
    + semantic_tasks + commonsense_tasks + comprehension_tasks + special_tasks

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


def tokenization(
    task_name, model_name,
    phase=["val"], k_shot=0,
    null=False, split_on_train=False,
    mismatched=False, hp_only=False,
    train_level="simple", val_level="hard"
):
    if split_on_train:
        py_io.write_json({
            "task": f"{task_name}",
            "paths": {
                "train": f"/content/tasks/curriculum/{task_name}/train.jsonl",
                "val": f"/content/tasks/curriculum/{task_name}/train.jsonl",
            },
            "name": f"{task_name}"
        }, f"/content/tasks/configs/{task_name}_config.json")
    elif mismatched:
        py_io.write_json({
            "task": f"{task_name}",
            "paths": {
                "train": f"/content/tasks/curriculum/{task_name}/train_{train_level}.jsonl",
                "val": f"/content/tasks/curriculum/{task_name}/val_{val_level}.jsonl",
            },
            "name": f"{task_name}"
        }, f"/content/tasks/configs/{task_name}_config.json")
    else:
        py_io.write_json({
            "task": f"{task_name}",
            "paths": {
                "train": f"/content/tasks/curriculum/{task_name}/train.jsonl",
                "val": f"/content/tasks/curriculum/{task_name}/val.jsonl",
            },
            "name": f"{task_name}"
        }, f"/content/tasks/configs/{task_name}_config.json")

    output_dir = f"./cache/{model_name}/{task_name}"
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"/content/tasks/configs/{task_name}_config.json",
        hf_pretrained_model_name_or_path=model_name,
        output_dir=output_dir,
        task_name=task_name,
        phases=phase,
        k_shot=k_shot,
        null=null,
        hp_only=hp_only
    ))


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


def train_configuration(
        task_name, model_name,
        cache_pth, val_task_key,
        classifier_type="linear",
        cross_task_name="None"):
    task_cache_base_path = cache_pth

    val_tasks = [task_name]
    if cross_task_name != "None":
        val_tasks = cross_task_name.split(',')

    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="/content/tasks/configs/",
        task_cache_base_path=task_cache_base_path,
        train_task_name_list=[task_name],
        val_task_name_list=val_tasks,
        train_batch_size=8,
        eval_batch_size=16,
        epochs=2,
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
    task_name,
    model_name,
    model_path,
    model_dir_name,
    model_load_mode="from_transformers",
    do_train=True,
    do_val=True,
    do_save_best=True,
    write_val_preds=True,
    freeze_encoder=False,
    k_shot=0,
    phase="main",
    mismatched=False,
    hp_only=False,
):
    if k_shot > 0:
        output_dir = f"./runs/{task_name}/{model_dir_name}/{k_shot}-shot"
        if phase == "null":
            output_dir = f"./runs/{task_name}/{model_dir_name}/{k_shot}-shot-null"
        elif mismatched or hp_only:
            output_dir = f"./runs/{task_name}/{model_dir_name}/{k_shot}-shot-{phase}"
    else:
        output_dir = f"./runs/{task_name}/{model_dir_name}/{phase}"
    os.makedirs(output_dir, exist_ok=True)

    if "probing" in phase:
        do_save_best = False
        write_val_preds = False

    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=f"./run_configs/{task_name}_run_config.json",
        output_dir=output_dir,
        hf_pretrained_model_name_or_path=model_name,
        model_load_mode=model_load_mode,
        model_path=model_path,
        model_config_path=f"./models/{model_name}/model/config.json",
        learning_rate=1e-5,
        eval_every_steps=500,
        do_train=do_train,
        do_val=do_val,
        do_save_best=do_save_best,
        write_val_preds=write_val_preds,
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
    "roberta-mnli": "roberta-large",
    "roberta-anli": "roberta-large",
    "roberta-anli-mix": "roberta-large",
    "deberta": "microsoft/deberta-base",
    "debertav3": "microsoft/deberta-v3-base",
}

MODEL_PATH_NAMES = {
    "bert1": "bert-base-uncased",
    "bert2": "bert-large-uncased",
    "roberta1": "roberta-base",
    "roberta1-glue": "roberta-base-glue",
    "roberta2": "roberta-large",
    "roberta-mnli": "roberta-large-mnli",
    "roberta-anli": "adversarial_nli_r3/roberta-large/main/",
    "roberta-anli-mix": "roberta-anli-mix",
    "deberta": "microsoft/deberta-base",
    "debertav3": "microsoft/deberta-v3-base",
}

MODEL_VAL_NAMES = {
    "bert1": "bert-base",
    "bert2": "bert-large",
    "roberta1": "roberta-base",
    "roberta2": "roberta-large",
    "roberta-mnli": "roberta-mnli",
    "roberta-anli-mix": "anli-mix-roberta",
    "roberta-anli": "roberta-anli",
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
    parser.add_argument("--probing_target", action="store_true",
                        help="enable probing phase"),
    parser.add_argument("--probing_base", action="store_true",
                        help="enable probing phase"),
    parser.add_argument("--null_baseline", action="store_true",
                        help="enable null baseline for v-entropy"),
    parser.add_argument("--hp_only", action="store_true",
                        help="enable hypothesis-only baseline"),
    parser.add_argument("--split_train", action="store_true",
                        help="enable difficulty split on train set"),
    parser.add_argument("--mismatched", action="store_true",
                        help="enable mismatched train and val sets on difficulty level"),
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
    parser.add_argument("--cross_task_name", type=str, default="None",
                        help="pre-trained transformer model name")
    parser.add_argument("--train_level", type=str, default="simple",
                        help="set the difficulty level for training")
    parser.add_argument("--val_level", type=str, default="hard",
                        help="set the difficulty level for validation")

    args = parser.parse_args()
    task_name = args.task_name
    exp_name = args.model_name
    model_name = MODEL_NAMES[args.model_name] if args.model_name in MODEL_NAMES else ""
    model_path_name = MODEL_PATH_NAMES[args.model_name] if args.model_name in MODEL_PATH_NAMES else ""

    print("Task Name: ", task_name)
    print("Model Name: ", model_name)
    print("Model Path Name: ", model_path_name)

    train_level = args.train_level
    val_level = args.val_level

    print(
        f"Train Distribution: {train_level} => Val Distribution: {val_level}")

    cache_path = f"./cache/{model_name}/"

    if args.tokenize:
        shutil.rmtree(f"{cache_path}/{task_name}", ignore_errors=True)
        if task_name == "curriculum":
            for task in CURRICULUM:
                tokenization(
                    task, model_name, phase="train",
                    k_shot=args.k_shot, null=args.null_baseline)
                tokenization(
                    task, model_name, phase="val",
                    k_shot=args.k_shot, null=args.null_baseline)
        else:
            tokenization(
                task_name, model_name,
                phase="train", k_shot=args.k_shot, null=args.null_baseline,
                split_on_train=args.split_train, mismatched=args.mismatched,
                hp_only=args.hp_only, train_level=train_level, val_level=val_level)
            tokenization(
                task_name, model_name,
                phase="val", k_shot=args.k_shot, null=args.null_baseline,
                split_on_train=args.split_train, mismatched=args.mismatched,
                hp_only=args.hp_only, train_level=train_level, val_level=val_level)
    elif args.tokenize_train:
        shutil.rmtree(f"{cache_path}/{task_name}/train", ignore_errors=True)
        if task_name == "curriculum":
            for task in CURRICULUM:
                tokenization(
                    task, model_name, phase="train",
                    k_shot=args.k_shot)
        else:
            tokenization(
                task_name, model_name,
                phase="train", k_shot=args.k_shot, null=args.null_baseline,
                split_on_train=args.split_train, mismatched=args.mismatched,
                hp_only=args.hp_only, train_level=train_level, val_level=val_level)
    elif args.tokenize_val:
        shutil.rmtree(f"{cache_path}/{task_name}/val", ignore_errors=True)
        shutil.rmtree(f"{cache_path}/{task_name}/val_labels",
                      ignore_errors=True)
        if task_name == "curriculum":
            for task in CURRICULUM:
                tokenization(task, model_name, phase="val", k_shot=args.k_shot)
        else:
            tokenization(
                task_name, model_name,
                phase="val", k_shot=args.k_shot, null=args.null_baseline,
                split_on_train=args.split_train, mismatched=args.mismatched,
                hp_only=args.hp_only, train_level=train_level, val_level=val_level)

    if args.setup_model:
        setup_model(model_name)

    if args.main_loop:

        load_mode = "from_transformers"
        do_train = True
        do_save_best = True
        write_val_preds = True

        model_val_name = MODEL_VAL_NAMES[args.model_name]
        if args.model_name == "roberta-anli":
            model_path = f"./runs/{model_path_name}/best_model.p"
            load_mode = "partial"
        else:
            model_path = f"./models/{model_path_name}/model/model.p"

        if args.load_best:
            load_mode = "all"
            do_train = False
            model_path = f"./runs/{task_name}/{model_val_name}/main/best_model.p"
            if args.k_shot > 0:
                model_path = f"./runs/{task_name}/{model_val_name}/{args.k_shot}-shot/best_model.p"
                if args.null_baseline:
                    model_path = f"./runs/{task_name}/{model_val_name}/{args.k_shot}-shot-null/best_model.p"
                elif args.hp_only:
                    model_path = f"./runs/{task_name}/{model_val_name}/{args.k_shot}-shot-hp/best_model.p"
                    do_save_best = False

        if "inference" in task_name:
            val_task_key = task_name.replace("_inference", "")
        else:
            val_task_key = "NONE"

        if args.k_shot < 1000 and args.k_shot > 0:
            do_save_best = False

        train_configuration(
            task_name,
            model_name,
            cache_pth=cache_path,
            val_task_key=val_task_key,
            classifier_type="linear",
            cross_task_name=args.cross_task_name,
        )

        phase = "main"
        freeze_encoder = False
        """if args.probing_base:
            phase = "probing_base"
            freeze_encoder = True
            load_mode = "partial"
        elif args.probing_target:
            phase = "probing_target"
            freeze_encoder = True
            load_mode = "partial"
            model_path = f"./runs/{task_name}/{model_val_name}/1000-shot/best_model.p"""
        if args.null_baseline:
            phase = "null"
            do_save_best = False
            write_val_preds = False
        elif args.mismatched:
            phase = f"{train_level}_{val_level}"
            do_save_best = False
            write_val_preds = False
        elif args.hp_only:
            phase = "hp"

        train(
            task_name=task_name,
            model_name=model_name,
            model_path=model_path,
            model_dir_name=model_val_name,
            model_load_mode=load_mode,
            do_train=do_train,
            do_save_best=do_save_best,
            write_val_preds=write_val_preds,
            freeze_encoder=freeze_encoder,
            k_shot=args.k_shot,
            phase=phase,
            mismatched=args.mismatched,
            hp_only=args.hp_only
        )


# python main_probing.py --main_loop --task_name monotonicity --exp_list bert2-mlp --exp_list roberta2 --exp_list roberta2-mlp --exp_list bert2
# python main_probing.py --tokenize --main_loop --task_name counterfactual --model roberta-anli --k_shot 10
