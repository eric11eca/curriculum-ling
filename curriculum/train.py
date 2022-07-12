import os

import jiant.utils.python.io as py_io
import jiant.utils.display as display
import jiant.proj.main.runscript as main_runscript
import jiant.proj.main.scripts.configurator as configurator


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
