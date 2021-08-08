import copy
import math

import torch
import torch.nn as nn
import torch.utils.data

from abc import ABC, abstractmethod
from util import device


class BaseModel(nn.Module, ABC):
    # pylint: disable=abstract-method
    name = 'base'

    def __init__(self):
        super().__init__()

        self.best_state_dict = None

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)

    def save(self, path):
        fname = self.get_name(path)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    @abstractmethod
    def get_args(self):
        pass

    @classmethod
    def load(cls, path):
        checkpoints = cls.load_checkpoint(path)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    @classmethod
    def load_checkpoint(cls, path):
        fname = cls.get_name(path)
        return torch.load(fname, map_location=device)

    @classmethod
    def get_name(cls, path):
        return '%s/model.tch' % (path)


class TransparentDataParallel(nn.DataParallel):

    def set_best(self, *args, **kwargs):
        return self.module.set_best(*args, **kwargs)

    def recover_best(self, *args, **kwargs):
        return self.module.recover_best(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self.module.save(*args, **kwargs)

    def train_batch(self, *args, **kwargs):
        return self.module.train_batch(*args, **kwargs)

    def eval_batch(self, *args, **kwargs):
        return self.module.eval_batch(*args, **kwargs)


class MLP(BaseModel):
    name = 'mlp'

    def __init__(self, task, embedding_size=768, n_classes=3, hidden_size=5,
                 nlayers=1, dropout=0.1, representation=None, n_words=None):
        super().__init__()

        # Save things to the model here
        self.dropout_p = dropout
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.n_classes = n_classes
        self.representation = representation
        self.n_words = n_words
        self.task = task

        if self.representation in ['onehot', 'random']:
            self.build_embeddings(n_words, embedding_size)

        self.mlp = self.build_mlp()
        self.out = nn.Linear(self.final_hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)

        self.criterion = nn.CrossEntropyLoss()

    def build_embeddings(self, n_words, embedding_size):
        if self.task == 'semgraph_edge':
            self.embedding_size = int(embedding_size / 2) * 2
            self.embedding = nn.Embedding(n_words, int(embedding_size / 2))
        else:
            self.embedding = nn.Embedding(n_words, embedding_size)

        if self.representation == 'random':
            self.embedding.weight.requires_grad = False

    def build_mlp(self):
        src_size = self.embedding_size
        tgt_size = self.hidden_size
        mlp = []
        for layer in range(self.nlayers):
            mlp += [nn.Linear(src_size, tgt_size)]
            mlp += [nn.ReLU()]
            mlp += [nn.Dropout(self.dropout_p)]
            src_size, tgt_size = tgt_size, int(tgt_size / 2)
        self.final_hidden_size = src_size
        return nn.Sequential(*mlp)

    def forward(self, x):
        if self.representation in ['onehot', 'random']:
            x = self.get_embeddings(x)

        x_emb = self.dropout(x)
        x = self.mlp(x_emb)
        logits = self.out(x)
        return logits

    def get_embeddings(self, x):
        x_emb = self.embedding(x)
        if len(x.shape) > 1:
            x_emb = x_emb.reshape(x.shape[0], -1)

        return x_emb

    def train_batch(self, data, target, optimizer):
        optimizer.zero_grad()
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target)
        loss.backward()
        optimizer.step()

        return loss.item() / math.log(2)

    def eval_batch(self, data, target):
        mlp_out = self(data)
        loss = self.criterion(mlp_out, target) / math.log(2)
        accuracy = (mlp_out.argmax(dim=-1) == target).float().detach().sum()
        loss = loss.item() * data.shape[0]

        return loss, accuracy

    def get_args(self):
        return {
            'nlayers': self.nlayers,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'dropout': self.dropout_p,
            'n_classes': self.n_classes,
            'representation': self.representation,
            'n_words': self.n_words,
            'task': self.task,
        }
