import shutil
import argparse

from curriculum.utils import (
    setup_model,
    MODEL_NAMES,
    HF_MODEL_PATH,
    MODEL_VAL_NAMES,
    ModelNotExsitError
)

from curriculum.tokenize import tokenization
from curriculum.train import train_configuration, train


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

    if args.model_name in MODEL_NAMES:
        model_name = MODEL_NAMES[args.model_name]
    else:
        raise ModelNotExsitError(args.model_name)

    if args.model_name in HF_MODEL_PATH:
        model_path_name = HF_MODEL_PATH[args.model_name]
    else:
        raise ModelNotExsitError(args.model_name)

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


# python main_probing.py --tokenize --main_loop --task_name counterfactual --model roberta-anli --k_shot 10
