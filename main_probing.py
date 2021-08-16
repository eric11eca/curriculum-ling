import os
import sys
import argparse
from types import prepare_class
import jiant.shared.caching as caching
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.runscript as main_runscript
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.export_model as export_model
import jiant.utils.display as display
import jiant.utils.python.io as py_io
sys.path.insert(0, "/content/jiant")


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


def tokenization(task_name, model_name, control=False, phase=["val"]):
    output_dir = f"./cache/{model_name}/{task_name}"
    if control:
        output_dir = f"./cache/control/{model_name}/{task_name}"
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"./tasks/configs/{task_name}_config.json",
        hf_pretrained_model_name_or_path=model_name,
        output_dir=output_dir,
        phases=phase,
    ))


def train_configuration(task_name, model_name, classifier_type, do_control=False):
    if do_control:
        task_cache_base_path = f"./cache/control/{model_name}/"
    else:
        task_cache_base_path = f"./cache/{model_name}/"
    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="/content/tasks/configs/",
        task_cache_base_path=task_cache_base_path,
        train_task_name_list=[task_name],
        val_task_name_list=[task_name],
        train_batch_size=8,
        eval_batch_size=16,
        epochs=30,
        num_gpus=1,
        classifier_type=classifier_type
    ).create_config()

    os.makedirs("./run_configs/", exist_ok=True)
    py_io.write_json(jiant_run_config,
                     f"./run_configs/{task_name}_run_config.json")
    display.show_json(jiant_run_config)


def train(task_name, model_name, model_path,
          do_train, freeze_encoder, model_dir_name):
    output_dir = f"./runs/{task_name}/{model_dir_name}/main"
    os.makedirs(output_dir, exist_ok=True)
    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=f"./run_configs/{task_name}_run_config.json",
        output_dir=output_dir,
        hf_pretrained_model_name_or_path=model_name,
        model_path=model_path,
        model_config_path=f"./models/{model_name}/model/config.json",
        learning_rate=1e-3,
        eval_every_steps=500,
        do_train=do_train,
        do_val=True,
        do_save=True,
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
    "roberta2": "roberta-large",
    "deberta": "microsoft/deberta-base",
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
    parser.add_argument("--task_name", type=str, default="semgraph2",
                        help="probing task name")
    parser.add_argument("--model_name", type=str, default="bert1",
                        help="pre-trained transformer model name")

    args = parser.parse_args()
    task_name = args.task_name
    exp_name = args.model_name
    model_name = MODEL_NAMES[args.model_name] if args.model_name in MODEL_NAMES else ""

    if len(args.train_pth) > 0 or len(args.val_pth) > 0:
        prepare_train_and_val_data(args.train_pth, args.val_pth)

    if args.tokenize:
        tokenization(task_name, model_name, phase="train")
        tokenization(task_name, model_name, phase="val")
    elif args.tokenize_train:
        tokenization(task_name, model_name, phase="train")
    elif args.tokenize_val:
        tokenization(task_name, model_name, phase="val")
    elif args.tokenize_control:
        tokenization(task_name, model_name, control=True, phase="val")

    if args.setup_model:
        setup_model(model_name)

    if args.main_loop:
        meta_config = py_io.read_json(
            f"./run_meta_configs/{task_name}_run_meta_configs.json")[exp_name]

        model_path = meta_config["model_pth"]
        model_name = meta_config["model_name"]
        model_val_name = meta_config["model_val_name"]

        do_train = meta_config["do_train"]
        do_control = meta_config["do_control"]
        freeze_encoder = meta_config["freeze_encoder"]
        classifier_type = meta_config["classifier_type"]

        train_configuration(task_name,
                            model_name,
                            classifier_type=classifier_type,
                            do_control=do_control)

        train(task_name=task_name,
              model_name=model_name,
              model_path=model_path,
              do_train=do_train,
              freeze_encoder=freeze_encoder,
              model_dir_name=model_val_name)
