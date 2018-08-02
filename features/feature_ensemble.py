#coding:utf8
import os
import argparse
import sys
sys.path.append("..")
import logging
    
import pandas as pd
import numpy as np

from common.utils import read_data, store_data
from conf.modelconf import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_interaction.txt')
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_interaction.txt')
        PHOTO_FEATURE_FILE = 'photo_feature' + '.' + fmt
        USER_FEATURE_FILE = 'user_feature' + '.' + fmt
        VISUAL_FEATURE_TRAIN_FILE = 'visual_feature_train' + '.' + fmt
        VISUAL_FEATURE_TEST_FILE = 'visual_feature_test' + '.' + fmt
        COMBINE_CTR_TRAIN_FEATURE_FILE = 'combine_ctr_feature_train' + '.' + fmt
        COMBINE_CTR_TEST_FEATURE_FILE = 'combine_ctr_feature_test' + '.' + fmt
        ONE_CTR_TRAIN_FEATURE_FILE = 'one_ctr_feature_train' + '.' + fmt
        ONE_CTR_TEST_FEATURE_FILE = 'one_ctr_feature_test' + '.' + fmt


    else:
        TRAIN_USER_INTERACT, offline_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, offline_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)

        PHOTO_FEATURE_FILE = 'photo_feature' + str(kfold) + '.' + fmt
        USER_FEATURE_FILE = 'user_feature' + str(kfold) + '.' + fmt
        VISUAL_FEATURE_TRAIN_FILE = 'visual_feature_train' + str(kfold) + '.' + fmt
        VISUAL_FEATURE_TEST_FILE = 'visual_feature_test' + str(kfold) + '.' + fmt
        COMBINE_CTR_TRAIN_FEATURE_FILE = 'combine_ctr_feature_train' + str(kfold) + '.' + fmt
        COMBINE_CTR_TEST_FEATURE_FILE = 'combine_ctr_feature_test' + str(kfold) + '.' + fmt
        ONE_CTR_TRAIN_FEATURE_FILE = 'one_ctr_feature_train' + str(kfold) + '.' + fmt
        ONE_CTR_TEST_FEATURE_FILE = 'one_ctr_feature_test' + str(kfold) + '.' + fmt



    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                                 sep='\t',
                                 header=None,
                                 names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])
    if args.online:
        user_item_test  = pd.read_csv(TEST_USER_INTERACT,
                                 sep='\t',
                                 header=None,
                                 names=['user_id', 'photo_id', 'time', 'duration_time'])
    else:
        user_item_test = pd.read_csv(TEST_USER_INTERACT,
                                      sep='\t',
                                      header=None,
                                      names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time',
                                             'duration_time'])



    photo_data = read_data(os.path.join(feature_store_dir, PHOTO_FEATURE_FILE), fmt)
    users = read_data(os.path.join(feature_store_dir, USER_FEATURE_FILE), fmt)

    combine_ctr_feats_train = read_data(os.path.join(feature_store_dir, COMBINE_CTR_TRAIN_FEATURE_FILE), fmt)
    combine_ctr_feats_test = read_data(os.path.join(feature_store_dir, COMBINE_CTR_TEST_FEATURE_FILE), fmt)
    one_ctr_feats_train = read_data(os.path.join(feature_store_dir, ONE_CTR_TRAIN_FEATURE_FILE), fmt)
    one_ctr_feats_test = read_data(os.path.join(feature_store_dir, ONE_CTR_TEST_FEATURE_FILE), fmt)

    path = os.path.join(feature_store_dir,VISUAL_FEATURE_TRAIN_FILE)
    if os.path.exists(path):
        visual_train = read_data(path,fmt)
    else:
        logging.warning('%s not exist, ignore.' % path)
        visual_train = None

    path = os.path.join(feature_store_dir, VISUAL_FEATURE_TEST_FILE)
    if os.path.exists(path):
        visual_test = read_data(path, fmt)
    else:
        logging.warning('%s not exist, ignore.' % path)
        visual_test = None
    
    user_item_train = pd.merge(user_item_train, users,
                          how='left',
                          on=['user_id'])
    
    user_item_train = pd.merge(user_item_train, photo_data,
                              how='left',
                              on=['photo_id'])

    user_item_train = pd.merge(user_item_train, combine_ctr_feats_train,
                               how='left', on=['user_id', 'photo_id'])

    user_item_train = pd.merge(user_item_train, one_ctr_feats_train,
                               how='left', on=['user_id', 'photo_id'])

    if visual_train is not None:
        user_item_train = pd.merge(user_item_train,visual_train,
                               how='left',on=['user_id','photo_id'])
    
    user_item_test = pd.merge(user_item_test, users,
                             how='left',
                             on=['user_id'])

    user_item_test = pd.merge(user_item_test, photo_data,
                          how='left',
                          on=['photo_id'])
    if visual_test is not None:
        user_item_test = pd.merge(user_item_test,visual_test,
                              how='left',
                              on=['user_id','photo_id'])
    user_item_test = pd.merge(user_item_test, combine_ctr_feats_test,
                               how='left', on=['user_id', 'photo_id'])

    user_item_test = pd.merge(user_item_test, one_ctr_feats_test,
                              how='left', on=['user_id', 'photo_id'])

    
    print(user_item_train.columns)


    user_item_train.fillna(0, inplace=True)
    user_item_test.fillna(0, inplace=True)
    user_item_train.sort_values(['user_id', 'photo_id'], inplace=True)
    user_item_test.sort_values(['user_id', 'photo_id'], inplace=True)

    input_features = id_features + user_features + photo_features + time_features + combine_ctr_features + one_ctr_features

    print(np.sum(user_item_train.isnull()))
    print(np.sum(user_item_test.isnull()))

    print(input_features)
    ensemble_train = user_item_train[input_features + y_label]

    if args.online:
        ensemble_test = user_item_test[input_features]
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + '.' + fmt
        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + '.' + fmt

    else:
        ensemble_test = user_item_test[input_features + y_label]
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + str(kfold) + '.' + fmt
        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + str(kfold) + '.' + fmt

    print(ensemble_train.info())
    print(ensemble_test.info())
    

    store_data(ensemble_train, os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)
    store_data(ensemble_test, os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
