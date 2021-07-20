import os
import sys
import argparse
import logging
import torch
import fasttext
import fasttext.util
from conllu import parse_incr
from scipy.sparse import lil_matrix
from transformers import BertTokenizer, BertModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--batch-size",
                        help="The size of the mini batches",
                        default=8,
                        required=False,
                        type=int)
    parser.add_argument("--language",
                        help="The language to use",
                        required=True,
                        type=str)
    parser.add_argument("--ud-path",
                        help="The path to raw ud data",
                        default='data/ud/ud-treebanks-v2.5/',
                        required=False,
                        type=str)
    parser.add_argument("--output-path",
                        help="The path to save processed data",
                        default='data/processed/',
                        required=False,
                        type=str)
    args = parser.parse_args()
    logging.info(args)

    return args


def load_fasttext():
    ft_path = '../data/fasttext'
    ft_fname = os.path.join(ft_path, 'cc.english.300.bin')
    if not os.path.exists(ft_fname):
        logging.info("Downloading fasttext model")
        temp_fname = fasttext.util.download_model(
            "english", if_exists='ignore')
        os.rename(temp_fname, ft_fname)
        os.rename(temp_fname + '.gz', ft_fname + '.gz')

    logging.info("Loading fasttext model")
    return fasttext.load_model(ft_fname)


def get_fasttext(fasttext_model, words):
    embeddings = [[fasttext_model[word] for word in sentence]
                  for sentence in words]
    return embeddings


def process_file(words, fasttext_model, output_file):
    logging.info("PHASE FOUR: getting fasttext embeddings")
    fast_embeddings = get_fasttext(fasttext_model, words)

    logging.info("PHASE FIVE: saving")
    output_data_raw = list(
        zip(bert_embeddings, fast_embeddings, all_ud, words))
    del bert_embeddings, fast_embeddings, all_ud, words

    # Prune the failed attempts:
    output_data = [(bert_embs, fast_embs, ud, words) for (
        bert_embs, fast_embs, ud, words) in output_data_raw if bert_embs != []]
    del output_data_raw
    output_ud = [(ud, words) for (_, _, ud, words) in output_data]
    output_bert = [(bert_embs, words)
                   for (bert_embs, _, _, words) in output_data]
    output_fast = [(fast_embeddings, words)
                   for (_, fast_embs, _, words) in output_data]
    del output_data

    # Pickle, compress, and save
    util.write_data(output_file % 'fast', output_fast)
    del output_fast

    logging.info("Completed {}".format(ud_file))


def process(language, ud_path, batch_size, bert_name, output_path):
    logging.info("Loading FastText Embedding")
    fasttext_model = load_fasttext(language)

    logging.info("Precessing language %s" % language)
    ud_file_base = get_ud_file_base(ud_path, language)
    output_file_base = get_data_file_base(output_path, language)
    for mode in ['train', 'dev', 'test']:
        ud_file = ud_file_base % mode
        output_file = output_file_base % (mode, '%s')
        process_file(bert_model, bert_tokenizer, fasttext_model,
                     batch_size, language, ud_file, output_file)

    logging.info("Process finished")


def main():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(processName)s : %(message)s', level=logging.INFO)
    args = get_args()

    batch_size = args.batch_size
    language = args.language
    ud_path = args.ud_path
    output_path = args.output_path
    bert_name = 'bert-base-multilingual-cased'

    with torch.no_grad():
        process(language, ud_path, batch_size, bert_name, output_path)


if __name__ == "__main__":
    main()
