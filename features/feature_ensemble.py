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
        VISUAL_FEATURE_FILE = 'visual_feature' + '.' + fmt


    else:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)

        PHOTO_FEATURE_FILE = 'photo_feature' + str(kfold) + '.' + fmt
        USER_FEATURE_FILE = 'user_feature' + str(kfold) + '.' + fmt
        VISUAL_FEATURE_FILE = 'visual_feature' + str(kfold) + '.' + fmt



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

    path = os.path.join(feature_store_dir,VISUAL_FEATURE_FILE)
    if os.path.exists(path):
        visual = read_data(os.path.join(feature_store_dir,VISUAL_FEATURE_FILE),fmt)
    else:
        logging.warning('%s not exist, ignore.' % path)
        visual = None
    
    user_item_train = pd.merge(user_item_train, users,
                          how='inner',
                          on=['user_id'])
    
    user_item_train = pd.merge(user_item_train, photo_data,
                              how='left',
                              on=['photo_id'])
    if visual is not None:
        user_item_train = pd.merge(user_item_train,visual,
                               how='left',on=['user_id','photo_id'])
    
    user_item_test = pd.merge(user_item_test, users,
                             how='inner',
                             on=['user_id'])

    user_item_test = pd.merge(user_item_test, photo_data,
                          how='left',
                          on=['photo_id'])
    if visual is not None:
        user_item_test = pd.merge(user_item_test,visual,
                              how='left',
                              on=['user_id','photo_id'])

    
    print(user_item_train.columns)
    user_item_train.fillna(0, inplace=True)
    user_item_test.fillna(0, inplace=True)
    input_features = id_features + user_features + photo_features + time_features
    
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


    for feature, dtype in feature_dtype_map.items():
        ensemble_train[feature] = ensemble_train[feature].astype(dtype)
        if feature in ensemble_test.columns:
            ensemble_test[feature] = ensemble_test[feature].astype(dtype)
    print(ensemble_train.info())
    print(ensemble_test.info())
    

    store_data(ensemble_train, os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)
    store_data(ensemble_test, os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
