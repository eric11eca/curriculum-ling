import torch
from dataset import SemgraphEdgeDataset
from dataset import MonotonicityDataset
from dataset import MultiWordSpanDataset

from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data_cls(task):
    if task in ["contradiction", "sentiment", "anaphora", "lexical", "relational", "paraphrase"]:
        return SemgraphEdgeDataset
    if task == 'monotonicity':
        return MonotonicityDataset
    if task == 'semgraph2':
        return SemgraphEdgeDataset


def generate_batch(batch):
    x = torch.cat([item[0].unsqueeze(0) for item in batch], dim=0)
    y = torch.cat([item[1].unsqueeze(0) for item in batch], dim=0)

    x, y = x.to(device), y.to(device, dtype=torch.long)
    return (x, y)


def get_data_loader(task, dataset_cls, representations,
                    pca_size, mode, batch_size, shuffle,
                    pca=None, classes=None, words=None):
    data_set = dataset_cls(task, representations, pca_size,
                           mode, pca=pca, classes=classes, words=words)
    dataloader = DataLoader(data_set, batch_size=batch_size,
                            shuffle=shuffle, collate_fn=generate_batch)
    return dataloader, data_set.pca, data_set.classes, data_set.words


def get_data_loaders(task, representations, pca_size, batch_size):
    dataset_cls = get_data_cls(task)

    trainloader, pca, classes, words = get_data_loader(task,
                                                       dataset_cls, representations, pca_size,
                                                       'train', batch_size=batch_size, shuffle=True)

    devloader, _, classes, words = get_data_loader(task,
                                                   dataset_cls, representations, pca_size,
                                                   'val', batch_size=batch_size, shuffle=False, pca=pca, classes=classes, words=words)

    return trainloader, devloader, devloader.dataset.n_classes, devloader.dataset.n_words
