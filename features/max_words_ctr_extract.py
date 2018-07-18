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
from conf.modelconf import *


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
args = parser.parse_args()


def gen_count_dict(data, labels, begin, end):
    total_dict = {}
    pos_dict = {}
    for i, d in enumerate(data):
        if i >= begin and i < end:
            continue
        for x in d:
            if x not in total_dict:
                total_dict[x] = 0.0
            if x not in pos_dict:
                pos_dict[x] = 0.0
            total_dict[x] += 1
            if labels[i] == True:
                pos_dict[x] += 1
    return total_dict, pos_dict

def count_word_ctr(train, test, labels):
    prior = alpha / (alpha + beta)
    total_dict, pos_dict = gen_count_dict(train, labels, 1, 0)
    train_res = []
    for i, d in enumerate(train):
        ctr_vec = []
        for x in d:
            if x not in total_dict:
                ctr_vec.append(prior)
            else:
                ctr_vec.append(alpha + pos_dict[x]/ (alpha+beta+total_dict[x]))
        if len(ctr_vec) == 0:
            train_res.append(prior)
        else:
            train_res.append(max(ctr_vec))

    test_res = []
    for i, d in enumerate(test):
        ctr_vec = []
        for x in d:
            if x not in total_dict:
                ctr_vec.append(prior)
            else:
                ctr_vec.append(alpha + pos_dict[x]/ (alpha+beta+total_dict[x]))
        if len(ctr_vec) == 0:
            test_res.append(prior)
        else:
            test_res.append(max(ctr_vec))

    return np.array(train_res), np.array(test_res)

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

    print(TRAIN_USER_INTERACT)
    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                                    sep='\t',
                                    usecols=[0, 1, 2],
                                    header=None,
                                    names=['user_id', 'photo_id','click'])

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

    p1 = set(text_train['photo_id'].unique())
    p2 = set(user_item_train['photo_id'].unique())
    print(len(p1), len(p2))
    print(len(p1 & p2))
    text_train['cover_words'] = text_train['cover_words'].apply(words_to_list)

    text_test['cover_words'] = text_test['cover_words'].apply(words_to_list)

    text_train['max_word_ctr'], text_test['max_word_ctr'] = count_word_ctr(text_train['cover_words'].values, text_test['cover_words'].values, user_item_train['click'].values)

    print(text_train.info())
    print(text_test.info())
    print(np.sum(text_train.isnull()))
    print(np.sum(text_test.isnull()))


    user_item_train = pd.merge(user_item_train, text_train, how='left', on=['photo_id'])
    user_item_test = pd.merge(user_item_test, text_test, how='left', on=['photo_id'])

    print(np.sum(user_item_train.isnull()))

    user_item_test.sort_values(['user_id', 'photo_id'], inplace=True)
    user_item_train.sort_values(['user_id', 'photo_id'], inplace=True)

    store_data(user_item_train[['max_word_ctr']], os.path.join(col_feature_store_dir, 'max_word_ctr_train.csv'),
               fmt)

    store_data(user_item_test[['max_word_ctr']], os.path.join(col_feature_store_dir, 'max_word_ctr_test.csv'),
               fmt)