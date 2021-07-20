import json
import string
import torch
import util
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.decomposition import PCA


class SemgraphNodeDataset(Dataset):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, representation, embedding_size, mode, pca=None, classes=None, words=None):
        self.mode = mode
        self.representation = representation
        self.embedding_size = embedding_size

        train_pth = '/content/tasks/data/semgraph2/train.jsonl'
        val_pth = '/content/tasks/data/semgraph2/val.jsonl'

        self.input_file_name = train_pth if mode == "train" else val_pth
        self.process(pca, classes, words)

        assert self.x.shape[0] == self.y.shape[0]
        self.n_instances = self.x.shape[0]

    def load_jsonline_data(self):
        data = []
        with open(self.input_file_name, 'r') as json_file:
            json_list = list(json_file)
            for json_str in json_list:
                result = json.loads(json_str)
                data.append(result)
        return data

    def process(self, pca, classes, words):
        if self.representation in ['fast']:
            self._process(pca, classes)
            self.words = words
            self.n_words = None
        else:
            self._process_index(classes, words)
            self.pca = pca

    def _process_index(self, classes, words):
        x_raw, y_raw = self.load_data_index()

        self.load_index(x_raw, words=words)
        self.load_classes(y_raw, classes=classes)

    def load_data_index(self):
        data_ud = util.read_data(self.input_name_base % (self.mode, 'ud'))

        x_raw, y_raw = [], []
        for sentence_ud, words in data_ud:
            for i, token in enumerate(sentence_ud):
                node_tag = token['tag']

                x_raw += [words[i]]
                y_raw += [node_tag]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)

        return x_raw, y_raw

    def load_index(self, x_raw, words=None):
        if words is None:
            # import ipdb; ipdb.set_trace()
            x, words = pd.factorize(x_raw, sort=True)
        else:
            new_words = set(x_raw) - set(words)
            if new_words:
                words = np.concatenate([words, list(new_words)])

            words_dict = {word: i for i, word in enumerate(words)}
            x = np.array([words_dict[token] for token in x_raw])

        self.x = torch.from_numpy(x)
        self.words = words

        self.n_words = len(words)

    def _process(self, pca, classes):
        x_raw, y_raw = self.load_data()

        self.load_embeddings(x_raw, pca=pca)
        self.load_classes(y_raw, classes=classes)

    def load_data(self):
        data_ud = util.read_data(self.input_name_base % (self.mode, 'ud'))
        data_embeddings = util.read_data(
            self.input_name_base % (self.mode, self.representation))

        x_raw, y_raw = [], []
        for (sentence_ud, words), (sentence_emb, _) in zip(data_ud, data_embeddings):
            for i, token in enumerate(sentence_ud):
                node_tag = token['tag']

                x_raw += [sentence_emb[i]]
                y_raw += [node_tag]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)

        return x_raw, y_raw

    def load_embeddings(self, x_raw, pca=None):
        pca_x = x_raw
        self.assert_size(pca_x)

        self.x = torch.from_numpy(pca_x)
        self.pca = pca

    def assert_size(self, x):
        assert len(x[0]) == self.embedding_size

    def load_classes(self, y_raw, classes=None):
        if self.mode != 'train':
            assert classes is not None

        if classes is None:
            y, classes = pd.factorize(y_raw, sort=True)
        else:
            new_classes = set(y_raw) - set(classes)
            if new_classes:
                classes = np.concatenate([classes, list(new_classes)])

            classes_dict = {pos_class: i for i,
                            pos_class in enumerate(classes)}
            y = np.array([classes_dict[token] for token in y_raw])

        self.y = torch.from_numpy(y)
        self.classes = classes

        self.n_classes = classes.shape[0]

    def tokenize(self, text):
        text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def __len__(self):
        return self.n_instances

    def __getitem__(self, index):
        return (self.x[index], self.y[index])


class SemgraphEdgeDataset(SemgraphNodeDataset):
    # pylint: disable=too-many-instance-attributes

    def load_data_index(self):
        data_ud = self.load_jsonline_data()

        x_raw, y_raw = [], []
        self.sentences = []
        for example in data_ud:
            for (target_num, target) in enumerate(example["targets"]):
                tokens = self.tokenize(example['text'])
                self.sentences.append(tokens)
                span1 = int(target["span1"][0])
                span2 = int(target["span2"][0])

                x_raw_tail = tokens[span1]
                x_raw_head = tokens[span2]

                x_raw += [[x_raw_tail, x_raw_head]]
                y_raw += [target["label"]]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)
        return x_raw, y_raw

    def load_index(self, x_raw, words=None):
        if words is None:
            words = []

        new_words = sorted(list(set(np.unique(x_raw)) - set(words)))
        if new_words:
            words = np.concatenate([words, new_words])

        words_dict = {word: i for i, word in enumerate(words)}
        x = np.array([[words_dict[token] for token in tokens]
                      for tokens in x_raw])

        self.x = torch.from_numpy(x)
        self.words = words

        self.n_words = len(words)

    def load_data(self):
        data_ud = self.load_jsonline_data()
        if self.mode == "train":
            data_embeddings = util.read_data("./dataset/output_fast_train")
        else:
            data_embeddings = util.read_data("./dataset/output_fast_val")

        x_raw, y_raw = [], []
        self.sentences = []
        for example, (sentence_emb, _) in zip(data_ud, data_embeddings):
            for (i, target) in enumerate(example["targets"]):
                span1 = int(target["span1"][0])
                span2 = int(target["span2"][0])

                x_raw_tail = sentence_emb[span1]
                x_raw_head = sentence_emb[span2]

                x_raw += [np.concatenate([x_raw_tail, x_raw_head])]
                y_raw += [target["label"]]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)
        return x_raw, y_raw
