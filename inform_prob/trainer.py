from model import Classifier, TransparentDataParallel
from train_info import TrainInfo
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import torch
import json
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(n_classes, n_words, embed, nlayers=1, hidden_size=128):
    mlp = Classifier(
        "semgraph_edge", embedding_size=600, n_classes=n_classes, hidden_size=hidden_size,
        nlayers=nlayers, dropout=0.5, representation=embed, n_words=n_words)

    if torch.cuda.device_count() > 1:
        mlp = TransparentDataParallel(mlp)
    return mlp.to(device)


def train(trainloader, devloader, model, eval_batches, wait_iterations):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device=device)

    with tqdm(total=wait_iterations) as pbar:
        mode_train_info = TrainInfo(pbar, wait_iterations, eval_batches)
        while not mode_train_info.finish:
            train_epoch(trainloader, devloader, model,
                        optimizer, criterion, mode_train_info)

    model.recover_best()


def _evaluate(evalloader, model):
    #criterion = nn.CrossEntropyLoss().to(device=device)
    dev_loss, dev_acc = 0, 0
    for x, y in evalloader:
        loss, acc = model.eval_batch(x, y)
        dev_loss += loss
        dev_acc += acc

    n_instances = len(evalloader.dataset)
    return {
        'loss': dev_loss / n_instances,
        'acc': dev_acc / n_instances
    }


def evaluate(evalloader, model):
    model.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model)
    model.train()
    return result


def train_epoch(trainloader, devloader, model, optimizer, criterion, mode_train_info):
    for x, y in trainloader:
        loss = model.train_batch(x, y, optimizer)
        mode_train_info.new_batch(loss)

        if mode_train_info.eval:
            dev_results = evaluate(devloader, model)

            if mode_train_info.is_best(dev_results):
                model.set_best()
            elif mode_train_info.finish:
                mode_train_info.print_progress(dev_results)
                return

            mode_train_info.print_progress(dev_results)


def eval_all(model, trainloader, devloader):
    train_results = evaluate(trainloader, model)
    dev_results = evaluate(devloader, model)

    train_loss = train_results['loss']
    test_loss = dev_results['loss']
    train_acc = train_results['acc']
    test_acc = dev_results['acc']

    print(f'Final loss. Train: {train_loss} Dev: {test_loss}')
    print(f'Final acc. Train: {train_acc} Dev: {test_acc}')
    return train_results, dev_results


def save_results(model, train_results, dev_results, results_fname):
    print("I am a new code")
    results = {'n_classes': model.n_classes,
               'embedding_size': model.embedding_size,
               'hidden_size': model.hidden_size,
               'nlayers': model.nlayers,
               'dropout_p': model.dropout_p,
               'train_loss': train_results['loss'].cpu().numpy().tolist(),
               'dev_loss': dev_results['loss'].cpu().numpy().tolist(),
               'train_acc': train_results['acc'].cpu().numpy().tolist(),
               'dev_acc': dev_results['acc'].cpu().numpy().tolist(),
               }
    with open(results_fname, "w") as write_file:
        json.dump(results, write_file, indent=4)


def save_checkpoints(task_name, emb_name, model, train_results, dev_results):
    checkpoint_dir = "checkpoints"+f"/{task_name}_{emb_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save(checkpoint_dir)
    results_fname = checkpoint_dir + '/results.json'
    save_results(model, train_results, dev_results, results_fname)
