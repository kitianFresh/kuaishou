# coding:utf8
import os
import argparse
import sys

sys.path.append('..')

import pandas as pd
import numpy as np
from common.utils import read_data, store_data
from conf.modelconf import get_data_file, data_dir, feature_dtype_map
from common.utils import count_combine_feat_ctr

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
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

    user_item_train = pd.merge(user_item_train, text_train, how='left', on=['photo_id'])
    user_item_test = pd.merge(user_item_test, text_test, how='left', on=['photo_id'])

    p1 = set(text_train['photo_id'].unique())
    p2 = set(user_item_train['photo_id'].unique())
    print(len(p1), len(p2))
    print(len(p1 & p2))

    user_item_train['max_user_word_ctr'], user_item_test['max_user_word_ctr'] = count_combine_feat_ctr(
                                                                            user_item_train['user_id'].astype(str).values,
                                                                            user_item_train['cover_words'].values,
                                                                            user_item_test['user_id'].astype(str).values,
                                                                           user_item_test['cover_words'].values,
                                                                           user_item_train['click'].values)



    if args.online:
        COMBINE_TRAIN_FEATURE_FILE = 'combine_ctr_feature_train' + '.' + fmt
        COMBINE_TEST_FEATURE_FILE = 'combine_ctr_feature_test' + '.' + fmt

    else:
        COMBINE_TRAIN_FEATURE_FILE = 'combine_ctr_feature_train' + str(kfold) + '.' + fmt
        COMBINE_TEST_FEATURE_FILE = 'combine_ctr_feature_test' + str(kfold) + '.' + fmt

    combine_train = user_item_train[['user_id', 'photo_id', 'max_user_word_ctr']]
    combine_test = user_item_test[['user_id', 'photo_id', 'max_user_word_ctr']]
    combine_train.sort_values(['user_id', 'photo_id'], inplace=True)
    combine_test.sort_values(['user_id', 'photo_id'], inplace=True)
    print(combine_train.info())
    print(combine_test.info())
    if not os.path.exists(feature_store_dir):
        os.mkdir(feature_store_dir)
    store_data(combine_train, os.path.join(feature_store_dir, COMBINE_TRAIN_FEATURE_FILE), fmt)
    store_data(combine_test, os.path.join(feature_store_dir, COMBINE_TEST_FEATURE_FILE), fmt)

    #column
    store_data(combine_train['max_user_word_ctr'], os.path.join(col_feature_store_dir, COMBINE_TRAIN_FEATURE_FILE), fmt)
    store_data(combine_test['max_user_word_ctr'], os.path.join(col_feature_store_dir, COMBINE_TRAIN_FEATURE_FILE), fmt)
