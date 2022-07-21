import os
import shutil
import argparse
import logging

from curriculum.utils import (
    setup_model,
    MODEL_NAMES,
    HF_MODEL_PATH,
    ModelNotExsitError
)

from curriculum.tokenize import tokenization
from curriculum.train import train_configuration, train

util_logger = logging.getLogger(
    'curriculum evaluation pipeline'
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./benchmark",
                        help="path to the benchmark data directory")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="path to the training output")
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
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="freeze the parameters of the pretrained encoder"),
    parser.add_argument("--task_name", type=str, default="semgraph2",
                        help="curriculum task name")
    parser.add_argument("--k_shot", type=int, default=0,
                        help="number of data shots")
    parser.add_argument("--exp_list", action='append',
                        help="curriculum experiment name")
    parser.add_argument("--model_name", type=str, default="bert1",
                        help="pre-trained transformer model name")
    parser.add_argument("--cross_task_name", type=str, default="None",
                        help="pre-trained transformer model name")
    parser.add_argument("--train_level", type=str, default="simple",
                        help="set the difficulty level for training")
    parser.add_argument("--val_level", type=str, default="hard",
                        help="set the difficulty level for validation")

    parser.add_argument("--num_epoch", type=int, default=3,
                        help="number of training epoches")
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="number of examples in a training batch")
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help="number of examples in a validation batch")
    parser.add_argument("--lr", type=int, default=1e-5, help="learning rate")

    args = parser.parse_args()
    task_name = args.task_name
    exp_name = args.model_name

    if args.model_name in MODEL_NAMES:
        model_base_name = MODEL_NAMES[args.model_name]
    else:
        raise ModelNotExsitError(args.model_name)

    if args.model_name in HF_MODEL_PATH:
        hf_model_name_or_path = HF_MODEL_PATH[args.model_name]
    else:
        raise ModelNotExsitError(args.model_name)

    if args.setup_model:
        util_logger.info(
            f"Download Huggingface pre-trained model {hf_model_name_or_path} as {args.model_name}")
        setup_model(args.model_name, hf_model_name_or_path)

    util_logger.info(f"Task Name: {task_name}")
    util_logger.info(f"Model Type: {model_base_name}")
    util_logger.info(
        f"Huggingface model name or path: {hf_model_name_or_path}")

    train_level = args.train_level
    val_level = args.val_level

    cache_path = f"./cache/{model_base_name}/"

    if args.tokenize:
        shutil.rmtree(f"{cache_path}/{task_name}", ignore_errors=True)
        tokenization(
            task_name,
            model_base_name,
            phase=["train", "val"],
            k_shot=args.k_shot,
            null=args.null_baseline,
            split_on_train=args.split_train,
            mismatched=args.mismatched,
            hp_only=args.hp_only,
            train_level=train_level,
            val_level=val_level
        )

    if args.main_loop:
        load_mode = "from_transformers"
        do_train = True
        do_save_best = True
        write_val_preds = True

        phase = "main"
        freeze_encoder = args.freeze_encoder

        if args.null_baseline:
            phase = "null"
            do_save_best = False
            write_val_preds = False
        elif args.mismatched:
            util_logger.info(
                f"Train Distribution: {train_level} => Val Distribution: {val_level}")
            phase = f"{train_level}_{val_level}"
            do_save_best = False
            write_val_preds = False
        elif args.hp_only:
            phase = "hp"
            do_save_best = False
            write_val_preds = False

        local_model_pth = f"./models/{args.model_name}/model/model.p"
        local_model_config_pth = f"./models/{args.model_name}/model/config.json"

        if args.load_best:
            load_mode = "all"
            do_train = False
            local_model_pth = f"./runs/{task_name}/{args.model_name}/{phase}"
            if args.k_shot > 0:
                local_model_pth = os.path.join(
                    local_model_pth, f"{args.k_shot}-shot")
            local_model_pth = os.path.join(local_model_pth, "best_model.p")

        if args.k_shot < 1000 and args.k_shot > 0:
            do_save_best = False

        train_configuration(
            task_name,
            data_dir=args.data_dir,
            cache_pth=cache_path,
            cross_task_name=args.cross_task_name,
        )

        train(
            task_name=task_name,
            output_dir=args.output_dir,
            model_pth=local_model_pth,
            model_config_pth=local_model_config_pth,
            hf_model_name=hf_model_name_or_path,
            model_dir_name=args.model_name,
            model_load_mode=load_mode,
            do_train=do_train,
            do_save_best=do_save_best,
            write_val_preds=write_val_preds,
            freeze_encoder=False,
            k_shot=args.k_shot,
            phase=phase
        )


# python main_probing.py --data_dir ./benchmark --task_name defeasible --model roberta-anli-mix --k_shot 1000 --main_loop --tokenize
