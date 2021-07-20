import os
import fasttext
import fasttext.util


def load_fasttext():
    ft_path = '../data/fasttext'
    ft_fname = os.path.join(ft_path, 'cc.en.300.bin')
    if not os.path.exists(ft_fname):
        print("Downloading fasttext model")
        temp_fname = fasttext.util.download_model(
            "en", if_exists='ignore')
        os.rename(temp_fname, ft_fname)
        os.rename(temp_fname + '.gz', ft_fname + '.gz')

    print("Loading fasttext model")
    return fasttext.load_model(ft_fname)


load_fasttext()
