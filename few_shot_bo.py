import os
import argparse
import logging
import numpy as np
import ray

import jiant.utils.display as display
import jiant.utils.python.io as py_io

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.runscript as main_runscript
import jiant.proj.main.scripts.configurator as configurator

import nevergrad as ng

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
#from ray.tune.suggest.dragonfly import DragonflySearch
from ray.tune.suggest.ax import AxSearch
from ray.tune.suggest.nevergrad import NevergradSearch


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
curriculum_dir = os.path.join(desktop, "research/curriculum-ling/")

curriculum_tasks = [
    "lexical", "transitive", "hypernymy", "hyponymy", "ner",
    "verbnet", "verbcorner", "syntactic_alternation", "syntactic_variation",
    "boolean", "comparative", "conditional", "counting", "negation", "quantifier",
    "monotonicity_infer", "syllogism",
    "sentiment", "kg_relations", "puns", "coreference", "context_align", "sprl",
    "entailment_tree", "proof_writer", "socialqa", "physicalqa", "atomic", "social_chem",
    "logiqa", "cosmoqa", "ester", "drop", "temporal", "spatial", "counterfactual"
]


def tokenization(
    task_name, model_name,
    phase=["train", "val"],
    k_shot=0
):
    py_io.write_json({
        "task": f"{task_name}",
        "paths": {
            "train": f"/content/tasks/curriculum/{task_name}/train.jsonl",
            "val": f"/content/tasks/curriculum/{task_name}/val.jsonl",
        },
        "name": f"{task_name}"
    }, f"/content/tasks/configs/{task_name}_config.json")

    output_dir = f"{curriculum_dir}/cache/{model_name}/{task_name}"
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_name=task_name,
        phases=phase,
        k_shot=0,
        output_dir=output_dir,
        task_config_path=f"/content/tasks/configs/{task_name}_config.json",
        hf_pretrained_model_name_or_path=model_name,
        null=False, hp_only=False
    ))


def get_k_shot_data_multi(train_lines, seed=42, k=10):
    np.random.seed(seed)
    np.random.shuffle(train_lines)
    label_list = {}
    for line in train_lines:
        label = line["gold_label"]
        if label not in label_list:
            label_list[label] = [line]
        else:
            label_list[label].append(line)
    new_train = []
    for label in label_list:
        for line in label_list[label][:k]:
            new_train.append(line)
    return new_train


def task_combinator(tasks, k_shot):
    train_data_collection = []
    val_data_collection = []

    for task in tasks:
        train_data = py_io.read_jsonl(
            f"/content/tasks/curriculum/{task}/train.jsonl")
        val_data = py_io.read_jsonl(
            f"/content/tasks/curriculum/{task}/val.jsonl")
        train_data_collection += get_k_shot_data_multi(train_data, k=k_shot)
        val_data_collection += get_k_shot_data_multi(val_data, k=k_shot)

    logger.info(f"Write {len(train_data_collection)} training data to file")
    task_name = "curriculum"
    os.makedirs("/content/tasks/configs/", exist_ok=True)
    os.makedirs(f"/content/tasks/curriculum/{task_name}", exist_ok=True)
    py_io.write_jsonl(
        data=train_data_collection,
        path=f"/content/tasks/curriculum/{task_name}/train.jsonl",
    )
    py_io.write_jsonl(
        data=val_data_collection,
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


def train_configuration(task_name, cache_pth):
    task_cache_base_path = cache_pth

    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="/content/tasks/configs/",
        task_cache_base_path=task_cache_base_path,
        train_task_name_list=[task_name],
        val_task_name_list=[task_name],
        train_batch_size=8,
        eval_batch_size=16,
        epochs=2,
        num_gpus=1,
        classifier_type="linear"
    ).create_config()

    os.makedirs(f"{curriculum_dir}/run_configs/", exist_ok=True)
    py_io.write_json(
        jiant_run_config,
        f"{curriculum_dir}/run_configs/{task_name}_run_config.json")
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
):
    if k_shot > 0:
        output_dir = f"{curriculum_dir}/runs/{task_name}/{model_dir_name}/{k_shot}-shot"
    else:
        output_dir = f"{curriculum_dir}/runs/{task_name}/{model_dir_name}/{phase}"
    os.makedirs(output_dir, exist_ok=True)

    if "probing" in phase:
        do_save_best = False
        write_val_preds = False

    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=f"{curriculum_dir}/run_configs/{task_name}_run_config.json",
        output_dir=output_dir,
        hf_pretrained_model_name_or_path=model_name,
        model_load_mode=model_load_mode,
        model_path=model_path,
        model_config_path=f"{curriculum_dir}/models/{model_name}/model/config.json",
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


def pretrain_function(model_name, task_selection, cache_path, k_shot):
    curriculum = [
        task for task in task_selection.keys()
        if task in curriculum_tasks and task_selection[task] >= 0.5
    ]

    logger.info("Current Task Combination")
    logger.info(curriculum)

    task_combinator(curriculum, k_shot=k_shot)
    tokenization("curriculum", model_name, k_shot=0)

    train_configuration(
        task_name="curriculum",
        cache_pth=cache_path
    )

    model_path = f"{curriculum_dir}/models/{model_name}/model/model.p"
    model_dir_name = "roberta-large"
    load_mode = "from_transformers"
    do_train = True
    do_save_best = True
    write_val_preds = False
    freeze_encoder = False

    train(
        task_name="curriculum",
        model_name=model_name,
        model_path=model_path,
        model_dir_name=model_dir_name,
        model_load_mode=load_mode,
        do_train=do_train,
        do_save_best=do_save_best,
        write_val_preds=write_val_preds,
        freeze_encoder=freeze_encoder,
        k_shot=0,
        phase="main"
    )


def objective(task_name, model_name, task_selection,
              k_shot_pre=2000, k_shot_tune=16):
    cache_path = f"{curriculum_dir}/cache/{model_name}/"
    pretrain_function(model_name, task_selection, cache_path, k_shot_pre)

    tokenization(task_name, model_name, k_shot=k_shot_tune)
    train_configuration(
        task_name=task_name,
        cache_pth=cache_path
    )

    model_path = f"{curriculum_dir}/runs/curriculum/{model_name}/main/best_model.p"
    model_dir_name = "roberta-large"
    load_mode = "partial"
    do_train = True
    do_save_best = True
    write_val_preds = False
    freeze_encoder = False

    train(
        task_name=task_name,
        model_name=model_name,
        model_path=model_path,
        model_dir_name=model_dir_name,
        model_load_mode=load_mode,
        do_train=do_train,
        do_save_best=do_save_best,
        write_val_preds=write_val_preds,
        freeze_encoder=freeze_encoder,
        k_shot=k_shot_tune,
        phase="main"
    )

    val_metrics = py_io.read_json(
        f"{curriculum_dir}/runs/{task_name}/{model_name}/{k_shot_tune}-shot/val_metrics.json")
    acc = val_metrics['aggregated']
    return acc


def training_function(config):
    logger.info("Generate New Combination...")
    logger.info(config)

    task_name = "adversarial_nli_r1"
    model_name = "roberta-large"
    k_shot_pre = 2000
    k_shot_tune = 16

    intermediate_score = objective(
        task_name, model_name, config,
        k_shot_pre, k_shot_tune
    )
    tune.report(objective=intermediate_score)


if __name__ == "__main__":
    ray.init(local_mode=True)
    """df_search = DragonflySearch(
        optimizer="bandit",
        domain="euclidean",
    )"""
    #df_search = ConcurrencyLimiter(df_search, max_concurrent=1)

    optimizer = ng.optimizers.ParametrizedOnePlusOne(
        mutation="doublefastga",
        noise_handling="optimistic"
    )
    algo = AxSearch()  # NevergradSearch(optimizer=optimizer)
    algo = ConcurrencyLimiter(algo, max_concurrent=1)
    scheduler = AsyncHyperBandScheduler()

    def trial_name_id(trial):
        return f"{trial.trainable_name}_{trial.trial_id}"

    logger.info("Tuning Loop Start: ...")
    analysis = tune.run(
        training_function,
        metric="objective",
        mode="max",
        name="ax",
        search_alg=algo,
        scheduler=scheduler,
        num_samples=50,
        resume=True,
        config={
            "coreference":      1,
            "kg_relations":     tune.uniform(0, 1),
            "sentiment":        tune.uniform(0, 1),
            "sprl":             tune.uniform(0, 1),
            "puns":             tune.uniform(0, 1),
            "context_align":    tune.uniform(0, 1),
            "entailment_tree":  tune.uniform(0, 1),
            "logiqa":           tune.uniform(0, 1),
            "cosmoqa":          tune.uniform(0, 1),
            "ester":            1,
            "drop":             tune.uniform(0, 1),
            "temporal":         tune.uniform(0, 1),
            "spatial":          tune.uniform(0, 1),
            "counterfactual":   tune.uniform(0, 1),
            "lexical":          1,
            "syntactic_variation": tune.uniform(0, 1),
            # "transitive":       tune.uniform(0, 1),
            # "hypernymy":        tune.uniform(0, 1),
            # "hyponymy":         tune.uniform(0, 1),
            # "ner":              tune.uniform(0, 1),
            # "verbnet":          tune.uniform(0, 1),
            # "verbcorner":       tune.uniform(0, 1),
            # "syntactic_alternation": tune.uniform(0, 1),
            # "monotonicity_infer": tune.uniform(0, 1),
            # "syllogism": tune.uniform(0, 1)
        },
        resources_per_trial={
            'cpu': 2,
            'gpu': 0,
        },
        trial_name_creator=trial_name_id
    )
    logger.info("Tuning Loop Completed!")
    print("Best hyperparameters found were: ", analysis.best_config)
