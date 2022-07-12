import jiant.utils.python.io as py_io
import jiant.proj.main.tokenize_and_cache as tokenize_and_cache


def tokenization(
    task_name,
    model_name,
    data_dir="./benchmark",
    phase=["val"],
    k_shot=0,
    null=False,
    split_on_train=False,
    mismatched=False,
    hp_only=False,
    train_level="simple",
    val_level="hard"
):
    if split_on_train:
        py_io.write_json({
            "task": f"{task_name}",
            "paths": {
                "train": f"{data_dir}/tasks/{task_name}/train.jsonl",
                "val": f"{data_dir}/tasks/{task_name}/train.jsonl",
            },
            "name": f"{task_name}"
        }, f"{data_dir}/configs/{task_name}_config.json")
    elif mismatched:
        py_io.write_json({
            "task": f"{task_name}",
            "paths": {
                "train": f"{data_dir}/tasks/{task_name}/train_{train_level}.jsonl",
                "val": f"{data_dir}/tasks/{task_name}/val_{val_level}.jsonl",
            },
            "name": f"{task_name}"
        }, f"{data_dir}/configs/{task_name}_config.json")
    else:
        py_io.write_json({
            "task": f"{task_name}",
            "paths": {
                "train": f"{data_dir}/tasks/{task_name}/train.jsonl",
                "val": f"{data_dir}/tasks/{task_name}/val.jsonl",
            },
            "name": f"{task_name}"
        }, f"{data_dir}/configs/{task_name}_config.json")

    output_dir = f"./cache/{model_name}/{task_name}"
    tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
        task_config_path=f"{data_dir}/configs/{task_name}_config.json",
        hf_pretrained_model_name_or_path=model_name,
        output_dir=output_dir,
        task_name=task_name,
        phases=phase,
        k_shot=k_shot,
        null=null,
        hp_only=hp_only
    ))
