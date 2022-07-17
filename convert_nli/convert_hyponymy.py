import random
from jiant.utils.python import io as py_io

from convert_nli.utils import count_labels, write_dataset_to_disk


def read_raw_data():
    train_data_raw = py_io.read_jsonl(
        "./benchmark/hyponymy/train.jsonl", encoding="utf8")
    val_data_raw = py_io.read_jsonl(
        "./benchmark/hyponymy/dev.jsonl", encoding="utf8")
    test_data_raw = py_io.read_jsonl(
        "./benchmark/hyponymy/test.jsonl", encoding="utf8")
    return train_data_raw, val_data_raw, test_data_raw


def convert_hyponymy_qa(data_raw):
    data = []
    for id, question in enumerate(data_raw):
        stem = question['question']['stem'].split("'")[1]
        premise = f"Here is a context, '{stem}'"
        term = question['notes']['surface_form']
        #premise = premise.replace(term, '<mask>')
        choices = question['question']['choices']
        answer = choices[int(question['answerKey'])]['text']
        choice_ids = [i for i in range(len(choices))]
        choice_ids.remove(int(question['answerKey']))
        wrong = choices[random.choice(choice_ids)]['text']

        # if id % 2 == 0:
        hypothesis = f"A specific type of {term} is {answer}, but not {wrong}."
        label = "entailed"
        data.append({
            "idx": id,
            "premise": premise,
            "hypothesis": hypothesis,
            "gold_label": label
        })
        # else:
        hypothesis = f"A specific type of {term} is {wrong}, but not {answer}."
        label = "not-entailed"
        data.append({
            "idx": id,
            "premise": premise,
            "hypothesis": hypothesis,
            "gold_label": label
        })
    return data


def convert():
    train_data_raw, val_data_raw, test_data_raw = read_raw_data()
    train_data_hyper = convert_hyponymy_qa(val_data_raw)
    val_data_hyper = convert_hyponymy_qa(train_data_raw + test_data_raw)

    print(count_labels(train_data_hyper))
    print(count_labels(val_data_hyper))

    write_dataset_to_disk(
        task_name="hyponymy",
        train_data=train_data_hyper,
        val_data=val_data_hyper
    )
