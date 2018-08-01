# coding:utf8
import os
import argparse
import sys

sys.path.append('..')

import pandas as pd
import numpy as np
from gensim import corpora, models
from collections import Counter
import fasttext
from common.utils import read_data, store_data
from text_cluster import train_cluster_model,cluster_model_predict
from word_embedding import load_model,train_word2vec
from conf.modelconf import get_data_file, data_dir, feature_dtype_map
from common.utils import count_feat_ctr
import logging
import bloscpack as bp


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
parser.add_argument('-c', '--cluster-nums', help='cluster nums', default=20)
args = parser.parse_args()




if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_text.txt',
                                                                                              args.online)
        TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_text.txt',
                                                                                             args.online)
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction.txt')
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_interaction.txt')


    else:
        TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_text' + str(kfold) + '.txt', online=False)
        TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_text' + str(kfold) + '.txt', online=False)

        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'test_interaction' + str(kfold) + '.txt', online=False)

    # trained once, use anywhere, shared for all online or offline folds
    DICTIONARY_PATH = os.path.join(data_dir, 'dictionary.txt')
    TFIDF_MODEL_PATH = os.path.join(data_dir, 'tfidf_model.model')
    WORD2VEC_MODEL_PATH = os.path.join(data_dir, 'word2vec.model')

    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                                  sep='\t',
                                  usecols=[0, 1, 2],
                                  header=None,
                                  names=['user_id', 'photo_id', 'click'])

    print(user_item_train.info())

    user_item_test = pd.read_csv(TEST_USER_INTERACT,
                                 sep='\t',
                                 usecols=[0, 1],
                                 header=None,
                                 names=['user_id', 'photo_id'])

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

    def words_to_list(words):
        if words == '0':
            return []
        else:
            return words.split(',')

    text_train['cover_words'] = text_train['cover_words'].apply(words_to_list)

    text_test['cover_words'] = text_test['cover_words'].apply(words_to_list)

    text_data = pd.concat([text_train, text_test])




    # 原始语料库
    corpus = []
    for cover_words in text_data['cover_words']:
        corpus.append(cover_words)

    word_embedding_corpus = []
    if os.path.exists(WORD2VEC_MODEL_PATH):
        model = load_model(WORD2VEC_MODEL_PATH)
        for sentence in corpus:
            word_embedding = np.zeros(100,)
            for word in sentence:
                try:
                    word_embedding += model.wv[word]
                except:
                    continue
            word_embedding_corpus.append(word_embedding)
    else:
        model = train_word2vec(corpus,WORD2VEC_MODEL_PATH)
        for sentence in corpus:
            word_embedding = np.zeros(100, )
            for word in sentence:
                try:
                    word_embedding += model.wv[word]
                except:
                    continue
            word_embedding_corpus.append(word_embedding)
    # word_embedding_corpus = np.array(word_embedding_corpus)

    print(type(word_embedding_corpus))


    cover_words_embedding_vector = word_embedding_corpus

    text_data.drop(['cover_words'], axis=1, inplace=True)

    # text_data['cover_words_tfidf'] = cover_words_tfidf_vector
    text_data['cover_words_embedding'] = cover_words_embedding_vector

    user_item_train = pd.merge(user_item_train, text_data, how='left', on=['photo_id'])
    user_item_test = pd.merge(user_item_test, text_data, how='left', on=['photo_id'])

    embedding_train = np.array(user_item_train[['cover_words_embedding']].values.tolist())
    embedding_test = np.array(user_item_test[['cover_words_embedding']].values.tolist())


    if args.online:
        EMBEDDING_TRAIN_FILE = 'cover_words_embedding_feature_train' + '.blp'
        EMBEDDING_TEST_FILE = 'cover_words_embedding_feature_test' + '.blp'

    else:
        EMBEDDING_TRAIN_FILE = 'cover_words_embedding_feature_train' + str(kfold) + '.blp'
        EMBEDDING_TEST_FILE = 'cover_words_embedding_feature_test' + str(kfold) + '.blp'


    bp.pack_ndarray_file(embedding_train, os.path.join(feature_store_dir, EMBEDDING_TRAIN_FILE))
    logging.info('matrix %s saved' % os.path.join(feature_store_dir, EMBEDDING_TRAIN_FILE))
    bp.pack_ndarray_file(embedding_test, os.path.join(feature_store_dir, EMBEDDING_TEST_FILE))
    logging.info('matrix %s saved' % os.path.join(feature_store_dir, EMBEDDING_TEST_FILE))


