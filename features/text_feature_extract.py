# coding:utf8
import os
import argparse
import sys

sys.path.append('..')

import pandas as pd
import numpy as np
from gensim import corpora, models
from collections import Counter

from common.utils import read_data, store_data

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
args = parser.parse_args()

if __name__ == '__main__':

    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'

    TRAIN_TEXT = '../sample/train_text.txt' if USE_SAMPLE else '../data/train_text.txt'
    TEST_TEXT = '../sample/test_text.txt' if USE_SAMPLE else '../data/test_text.txt'
    text_train = pd.read_csv(TRAIN_TEXT,
                             sep='\t',
                             header=None,
                             names=['photo_id', 'cover_words'])

    print(text_train.info())

    text_test = pd.read_csv(TEST_TEXT,
                            sep='\t',
                            header=None,
                            names=['photo_id', 'cover_words'])

    print(text_test.info())

    text_data = pd.concat([text_train, text_test])


    def words_to_list(words):
        if words == '0':
            return []
        else:
            return words.split(',')


    text_data['cover_words'] = text_data['cover_words'].apply(words_to_list)
    text_data['cover_length'] = text_data['cover_words'].apply(lambda words: len(words))

    text_data.fillna(0, inplace=True)

    # 原始语料库
    corpus = []
    for cover_words in text_data['cover_words']:
        corpus.append(cover_words)

    # 词典
    DICTIONARY_PATH = '../sample/dictionary.txt' if USE_SAMPLE else '../data/dictionary.txt'
    if not os.path.exists(DICTIONARY_PATH):
        dictionary = corpora.Dictionary(corpus)
        dictionary.save(DICTIONARY_PATH)
    else:
        dictionary = corpora.Dictionary.load(DICTIONARY_PATH)

    corpus = [dictionary.doc2bow(text) for text in corpus]

    # tfidf模型
    TFIDF_MODEL_PATH = '../sample/tfidf_model.model' if USE_SAMPLE else '../data/tfidf_model.model'
    if not os.path.exists(TFIDF_MODEL_PATH):
        tfidf_model = models.TfidfModel(corpus)
        tfidf_model.save(TFIDF_MODEL_PATH)
    else:
        tfidf_model = models.TfidfModel.load(TFIDF_MODEL_PATH)
    corpus_tfidf = tfidf_model[corpus]

    key_words = []
    for item in corpus_tfidf:
        word_tfidf = sorted(item, key=lambda x: x[1], reverse=True)
        if len(word_tfidf) >= 2:
            key_words.extend([dictionary.get(i[0]) for i in word_tfidf[:2]])
        elif len(word_tfidf) == 1:
            key_words.extend([dictionary.get(word_tfidf[0][0])])

    count = Counter(key_words)
    print(len(count))
    # top_k关键词
    top_key_words = count.most_common(2000)
    top_key_words = [i[0] for i in top_key_words]


    def key_words_num(words):
        num = 0
        for word in words:
            if word in top_key_words:
                num += 1
        return num


    text_data['key_words_num'] = text_data['cover_words'].apply(key_words_num)
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_df=0.7)
    corpus = text_data['cover_words'].apply(lambda words: ' '.join(words))
    tfidf = vectorizer.fit_transform(corpus)
    avg_tfidf = np.mean(tfidf, axis=1)
    text_data['avg_tfidf'] = avg_tfidf

    text_data.drop(['cover_words'], axis=1, inplace=True)
    TEXT_FEATURE_FILE = 'text_feature'
    TEXT_FEATURE_FILE = TEXT_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else TEXT_FEATURE_FILE + '.' + fmt
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)
    store_data(text_data, os.path.join(feature_store_path, TEXT_FEATURE_FILE), fmt)
