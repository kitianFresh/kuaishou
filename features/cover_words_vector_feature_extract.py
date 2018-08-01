# coding:utf8
import os
import argparse
import sys
import gc

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append('..')

# import modin.pandas as pd
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from conf.modelconf import *
from common.utils import read_data


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
parser.add_argument('-m', '--max-feature', help='max feature to use count vector', default=20000)
args = parser.parse_args()



if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)

    if args.online:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction.txt')
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_interaction.txt')
        TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_text.txt',
                                                                                              args.online)
        TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_text.txt',
                                                                                             args.online)


    else:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)
        TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_text' + str(kfold) + '.txt', online=False)
        TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_text' + str(kfold) + '.txt', online=False)


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


    text_train['cover_words'] = text_train['cover_words'].apply(words_to_list)
    text_test['cover_words'] = text_test['cover_words'].apply(words_to_list)
    text_train['cover_words'] = text_train['cover_words'].apply(lambda x: ' '.join(x))
    text_test['cover_words'] = text_test['cover_words'].apply(lambda  x: ' '.join(x))

    user_item_train = pd.merge(user_item_train, text_train, how='left', on=['photo_id'])
    user_item_test = pd.merge(user_item_test, text_test, how='left', on=['photo_id'])

    del text_train
    del text_test
    gc.collect()

    user_item_train.reset_index(drop=True, inplace=True)
    user_item_test.reset_index(drop=True, inplace=True)


    print(user_item_train.info())
    print(user_item_test.info())


    user_item_train.sort_values(['user_id', 'photo_id'], inplace=True)
    user_item_test.sort_values(['user_id', 'photo_id'], inplace=True)
    print(user_item_train.head())
    print(user_item_test.head())

    cv = CountVectorizer(token_pattern='\w+', max_features=int(args.max_feature))  # max_features = 20000
    print("开始cv.....")
    # pos_user_id, neg_user_id
    print("生成 "  + "cover_words CountVector")
    feature = "cover_words"
    cv.fit(user_item_train[feature].astype('str'))
    print("开始转换 " + feature + " CountVector")
    train_temp = cv.transform(user_item_train[feature].astype('str'))
    test_temp = cv.transform(user_item_test[feature].astype('str'))

    print(feature + " is over")

    all_train_data_x = train_temp
    test_data_x = test_temp

    print('cv prepared')
    print(all_train_data_x.shape)
    print(test_data_x.shape)

    for feat in id_features:
        if args.online:
            sparse.save_npz(os.path.join(feature_store_dir, 'cover_words_vector_feature_train.npz'), all_train_data_x)
            sparse.save_npz(os.path.join(feature_store_dir, 'cover_words_vector_feature_test.npz'), test_data_x)
        else:
            sparse.save_npz(os.path.join(feature_store_dir, 'cover_words_vector_feature_train' + str(kfold) + '.npz'), all_train_data_x)
            sparse.save_npz(os.path.join(feature_store_dir, 'cover_words_vector_feature_test' + str(kfold) + '.npz'), test_data_x)


