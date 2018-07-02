#coding:utf8

import os
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
sys.path.append("..")
    
import pandas as pd
import numpy as np

from common.utils import read_data, store_data
from conf.modelconf import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-p', '--pool-type', help='pool type, threads or process, here use process for more performance')
parser.add_argument('-n', '--num-workers', help='workers num in pool')
parser.add_argument('-c', '--column-split', help='column to split', action='append')
parser.add_argument('-t', '--table', help='original table used to split')

args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv' 
    pool_type = args.pool_type if args.pool_type else 'thread'
    n = args.num_workers if args.num_workers else 8
    split_features = args.column_split if args.column_split else []
    table = args.table if args.table else None

    Executor = ThreadPoolExecutor if pool_type == 'thread' else ProcessPoolExecutor
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)
        
    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'
    if not os.path.exists(col_feature_store_path):
        os.mkdir(col_feature_store_path)
        
    TRAIN_USER_INTERACT = '../sample/train_interaction.txt' if USE_SAMPLE else '../data/train_interaction.txt'
    TEST_INTERACT = '../sample/test_interaction.txt' if USE_SAMPLE else '../data/test_interaction.txt'

    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                             sep='\t',
                             header=None,
                             names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])
    user_item_test = pd.read_csv(TEST_INTERACT,
                             sep='\t',
                             header=None,
                             names=['user_id', 'photo_id', 'time', 'duration_time'])

    PHOTO_FEATURE_FILE = 'photo_feature'
    PHOTO_FEATURE_FILE = PHOTO_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else PHOTO_FEATURE_FILE + '.' + fmt
    photo_data = read_data(os.path.join(feature_store_path, PHOTO_FEATURE_FILE), fmt)


    USER_FEATURE_FILE = 'user_feature'
    USER_FEATURE_FILE = USER_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else USER_FEATURE_FILE +  '.' + fmt
    users = read_data(os.path.join(feature_store_path, USER_FEATURE_FILE), fmt)

    VISUAL_FEATURE_FILE = 'visual_feature'
    VISUAL_FEATURE_FILE = VISUAL_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else VISUAL_FEATURE_FILE + '.' + fmt
    visual = read_data(os.path.join(feature_store_path,VISUAL_FEATURE_FILE),fmt)

    user_item_train = pd.merge(user_item_train, users,
                          how='inner',
                          on=['user_id'])

    user_item_train = pd.merge(user_item_train, photo_data,
                              how='left',
                              on=['photo_id'])

    user_item_train = pd.merge(user_item_train,visual,
                               how='left',on=['user_id','photo_id'])

    user_item_test = pd.merge(user_item_test, users,
                             how='inner',
                             on=['user_id'])

    user_item_test = pd.merge(user_item_test, photo_data,
                          how='left',
                          on=['photo_id'])

    user_item_test = pd.merge(user_item_test,visual,
                              how='left',
                              on=['user_id','photo_id'])


    print(user_item_train.columns)
    user_item_train.fillna(0, inplace=True)
    user_item_test.fillna(0, inplace=True)


    input_features = id_features + user_features + photo_features + time_features
    ensemble_train = user_item_train[input_features + y_label]
    ensemble_test = user_item_test[input_features]


    print(input_features)
    for feat in input_features + y_label:
        ensemble_train[feat] = ensemble_train[feat].astype(feature_dtype_map.get(feat))
    print(ensemble_train.info())

    for feat in input_features:
        ensemble_test[feat] = ensemble_test[feat].astype(feature_dtype_map.get(feat))
    print(ensemble_test.info())
#     fmt = 'h5'

    input_features = input_features if len(split_features) == 0 else split_features
    tasks_args = []
    for feature in set(input_features) - set(id_features):
        train_file = feature  + '_train' + '.' + fmt
        test_file = feature  + '_test' + '.' + fmt
        args = (ensemble_train[id_features+[feature]], os.path.join(col_feature_store_path, train_file), fmt)
        tasks_args.append(args)
        args = (ensemble_test[id_features+[feature]], os.path.join(col_feature_store_path, test_file), fmt)
        tasks_args.append(args)

    if len(split_features) == 0:
        feature = y_label[0]
        train_file = feature  + '_train' + '.' + fmt
        args = (ensemble_train[id_features+[feature]], os.path.join(col_feature_store_path, train_file), fmt)
        tasks_args.append(args)

    def feature_saver(args):
        df, path, fmt = args
        res = store_data(df, path, fmt)
        return res
    start_time_1 = time.clock()
    with Executor(max_workers=n) as executor:
        for file in executor.map(feature_saver,  tasks_args):
            print('%s saved' % file)
    print ("%s pool execution in %s seconds" % (pool_type, str(time.clock() - start_time_1)))
