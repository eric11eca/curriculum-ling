import jiant.shared.caching as caching
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.runscript as main_runscript
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.export_model as export_model
import jiant.utils.display as display
import jiant.utils.python.io as py_io
import os
import sys
sys.path.insert(0, "/content/jiant")


def train_configuration(task_name):
    jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path="./tasks/configs",
        task_cache_base_path="./cache",
        train_task_name_list=[task_name],
        val_task_name_list=[task_name],
        train_batch_size=8,
        eval_batch_size=16,
        epochs=3,
        num_gpus=1,
    ).create_config()

    os.makedirs("./run_configs/", exist_ok=True)
    py_io.write_json(jiant_run_config,
                     f"./run_configs/{task_name}_run_config.json")
    display.show_json(jiant_run_config)


def train(task_name, model_name):
    run_args = main_runscript.RunConfiguration(
        jiant_task_container_config_path=f"./run_configs/{task_name}_run_config.json",
        output_dir=f"./runs/{task_name}",
        hf_pretrained_model_name_or_path=model_name,
        model_path=f"./models/{model_name}/model/model.p",
        model_config_path=f"./models/{model_name}/model/config.json",
        learning_rate=1e-5,
        eval_every_steps=500,
        do_train=True,
        do_val=True,
        do_save=True,
        force_overwrite=True,
        no_cuda=False
    )
    main_runscript.run_loop(run_args)


def tokenization(task_name, model_name):
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"./tasks/configs/{task_name}_config.json",
        hf_pretrained_model_name_or_path=model_name,
        output_dir=f"./cache/{task_name}",
        phases=["train", "val"],
    ))


def setup_model(model_name, exist=True):
    if not exist:
        export_model.export_model(
            hf_pretrained_model_name_or_path=model_name,
            output_base_path=f"./models/{model_name}",
        )


if __name__ == "__main__":
    task_name = "semgraph2"
    bert = "bert-base-uncased"
    roberta = "roberta-base"

    setup_model(bert, True)

    print("Setup Jiant Run_Configurations: ")
    train_configuration(task_name)

    print("Jiant Training Session Starts: ")
    train(task_name=task_name, model_name=bert)
