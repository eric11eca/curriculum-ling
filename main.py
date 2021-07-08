import logging as log

log.basicConfig(
    format="%(asctime)s: %(message)s", datefmt="%m/%d %I:%M:%S %p", level=log.INFO
)  # noqa
import argparse
import glob
import io
import os
import random
import subprocess
import sys
import time
import copy
import torch


def initial_setup(args):
    """
    Sets up email hook, creating seed, and cuda settings.
    Parameters
    ----------------
    args: Params object
    Returns
    ----------------
    tasks: list of Task objects
    pretrain_tasks: list of pretraining tasks
    target_tasks: list of target tasks
    vocab: list of vocab
    word_embs: loaded word embeddings, may be None if args.input_module in
    {gpt, elmo, elmo-chars-only, bert-*}
    model: a MultiTaskModel object
    """

    log_fh = log.FileHandler("./Logs/")
    log_fmt = log.Formatter("%(asctime)s: %(message)s",
                            datefmt="%m/%d %I:%M:%S %p")
    log_fh.setFormatter(log_fmt)
    log.getLogger().addHandler(log_fh)

    # config_file = os.path.join(args.run_dir, "params.conf")
    # config.write_params(args, config_file)

    print_args = select_relevant_print_args(args)
    log.info("Parsed args: \n%s", print_args)

    log.info("Saved config to %s", config_file)

    seed = random.randint(
        1, 10000) if args.random_seed < 0 else args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    log.info("Using random seed %d", seed)

    try:
        if not torch.cuda.is_available():
            raise EnvironmentError(
                "CUDA is not available")
        log.info("Using GPU %d", 0)
        torch.cuda.set_device(0)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        log.warning(
            "GPU access failed. You might be using a CPU-only installation of PyTorch. "
            "Falling back to CPU."
        )

    return args, seed


def main(args):
    """ Train a model for multitask-training."""

    # Check for deprecated arg names
    args, seed = initial_setup(args)
    # Load tasks
    log.info("Loading tasks...")
    start_time = time.time()
    pretrain_tasks, target_tasks, vocab, word_embs = build_tasks(args)
    tasks = sorted(set(pretrain_tasks + target_tasks), key=lambda x: x.name)
    log.info("\tFinished loading tasks in %.3fs", time.time() - start_time)
    log.info("\t Tasks: {}".format([task.name for task in tasks]))

    # Build model
    log.info("Building model...")
    start_time = time.time()
    model = build_model(args, vocab, word_embs, tasks)
    log.info("Finished building model in %.3fs", time.time() - start_time)

    check_configurations(args, pretrain_tasks, target_tasks)
    if args.do_pretrain:
        # Train on pretrain tasks
        log.info("Training...")
        stop_metric = pretrain_tasks[0].val_metric if len(
            pretrain_tasks) == 1 else "macro_avg"
        should_decrease = (
            pretrain_tasks[0].val_metric_decreases if len(
                pretrain_tasks) == 1 else False
        )
        trainer, _, opt_params, schd_params = build_trainer(
            args, [], model, args.run_dir, should_decrease, phase="pretrain"
        )
        to_train = [(n, p)
                    for n, p in model.named_parameters() if p.requires_grad]
        _ = trainer.train(
            pretrain_tasks,
            stop_metric,
            args.batch_size,
            args.weighting_method,
            args.scaling_method,
            to_train,
            opt_params,
            schd_params,
            args.load_model,
            phase="pretrain",
        )

    # For checkpointing logic
    if not args.do_target_task_training:
        strict = True
    else:
        strict = False

    if args.do_target_task_training:
        # Train on target tasks
        pre_target_train_path = setup_target_task_training(
            args, target_tasks, model, strict)
        target_tasks_to_train = copy.deepcopy(target_tasks)
        # Check for previous target train checkpoints
        task_to_restore, _, _ = check_for_previous_checkpoints(
            args.run_dir, target_tasks_to_train, "target_train", args.load_model
        )
        if task_to_restore is not None:
            # If there is a task to restore from, target train only on target tasks
            # including and following that task.
            last_task_index = [task.name for task in target_tasks_to_train].index(
                task_to_restore)
            target_tasks_to_train = target_tasks_to_train[last_task_index:]
        for task in target_tasks_to_train:
            # Skip tasks that should not be trained on.
            if task.eval_only_task:
                continue

            params_to_train = load_model_for_target_train_run(
                args, pre_target_train_path, model, strict, task
            )
            trainer, _, opt_params, schd_params = build_trainer(
                args,
                [task.name],
                model,
                args.run_dir,
                task.val_metric_decreases,
                phase="target_train",
            )

            _ = trainer.train(
                tasks=[task],
                stop_metric=task.val_metric,
                batch_size=args.batch_size,
                weighting_method=args.weighting_method,
                scaling_method=args.scaling_method,
                train_params=params_to_train,
                optimizer_params=opt_params,
                scheduler_params=schd_params,
                load_model=(task.name == task_to_restore),
                phase="target_train",
            )

    if args.do_full_eval:
        log.info("Evaluating...")
        splits_to_write = evaluate.parse_write_preds_arg(args.write_preds)

        # Evaluate on target_tasks.
        for task in target_tasks:
            # Find the task-specific best checkpoint to evaluate on.
            task_to_use = model._get_task_params(
                task.name).get("use_classifier", task.name)
            ckpt_path = get_best_checkpoint_path(args, "eval", task_to_use)
            assert ckpt_path is not None
            load_model_state(model, ckpt_path, args.cuda,
                             skip_task_models=[], strict=strict)
            evaluate_and_write(args, model, [task], splits_to_write)

    log.info("Done!")
