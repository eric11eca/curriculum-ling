import numpy as np
import bz2
import random
import pickle
import pathlib
import shutil
import csv
import io
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LANGUAGE_CODES = {
    'english': 'en',
    'czech': 'cs',
    'basque': 'eu',
    'finnish': 'fi',
    'turkish': 'tr',
    'tamil': 'ta',
    'korean': 'ko',
    'marathi': 'mr',
    'urdu': 'ur',
    'telugu': 'te',
    'indonesian': 'id',
}


def config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(filename, results):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def write_data(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def read_data(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def rmdir_if_exists(fdir):
    if os.path.exists(fdir):
        shutil.rmtree(fdir)


def file_len(fname):
    if not os.path.isfile(fname):
        return 0

    with open(fname, 'r') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
