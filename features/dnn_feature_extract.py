
# coding:utf8
import os
import argparse
import sys
sys.path.append("..")
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import pandas as pd
import numpy as np

from common.utils import read_data, store_data, FeatureMerger, normalize_min_max
from conf.deep_conf import *
from conf.modelconf import online_data_dir, offline_data_dir, id_features, y_label
from common import utils

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
parser.add_argument('-a', '--all', help='how to read ensemble table,  all or merge by columns',action='store_true')
parser.add_argument('-n', '--num-workers', help='num used to merge columns', default=cpu_count())

args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    all_one = args.all
    num_workers = args.num_workers

    if args.online:
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + '.' + fmt
        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + '.' + fmt

    else:
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + str(kfold) + '.' + fmt
        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + str(kfold) + '.' + fmt

    if args.online:
        feature_store_dir = os.path.join(online_data_dir, 'features')
        col_feature_store_dir = os.path.join(feature_store_dir, 'columns')

        start = time.time()
        if all_one:
            ensemble_train = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)
            ensemble_test = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
        else:
            feature_to_use = id_features + features_to_train
            fm_trainer = FeatureMerger(col_feature_store_dir, feature_to_use + y_label, fmt=fmt, data_type='train',
                                       pool_type='process', num_workers=num_workers)
            fm_tester = FeatureMerger(col_feature_store_dir, feature_to_use, fmt=fmt, data_type='test',
                                      pool_type='process', num_workers=num_workers)
            ensemble_train = fm_trainer.concat()
            ensemble_test = fm_tester.concat()
        end = time.time()
        print('data read in %s seconds' % str(end - start))

    else:
        feature_store_dir = os.path.join(offline_data_dir, 'features')
        col_feature_store_dir = os.path.join(feature_store_dir, 'columns')

        start = time.time()
        if all_one:
            ensemble_train = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)
            ensemble_test = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
        else:
            feature_to_use = id_features + features_to_train
            fm_trainer = FeatureMerger(col_feature_store_dir, feature_to_use + y_label, fmt=fmt, data_type='train',
                                       pool_type='process', num_workers=num_workers)
            fm_tester = FeatureMerger(col_feature_store_dir, feature_to_use + y_label, fmt=fmt, data_type='test',
                                      pool_type='process', num_workers=num_workers)
            ensemble_train = fm_trainer.concat()
            ensemble_test = fm_tester.concat()
        end = time.time()
        print('data read in %s seconds' % str(end - start))

    print(ensemble_train.info())
    print(ensemble_test.info())

    num_train, num_test = ensemble_train.shape[0], ensemble_test.shape[0]
    ensemble_data = pd.concat(
        [ensemble_train[id_features + features_to_train], ensemble_test[id_features + features_to_train]])


    cates = []
    for col in to_discretization_features:
        name = col + '_discretization'
        func = getattr(utils, name) if hasattr(utils, name) else None
        if func is not None and callable(func):
            print(func.__name__)
            ensemble_data[col] = ensemble_data[col].apply(func)
            cates.append(col)

    ensemble_data[cates] = ensemble_data[cates].astype(np.int8)


    numeric_features = float32_cols
    normalize_min_max(ensemble_data, numeric_features)

    # to do....
    # real_value_vector_features = ['cover_words', 'visual', 'pos_photo_id', 'neg_photo_id', 'pos_photo_cluster_label', 'neg_photo_cluster_label', 'pos_user_id', 'neg_user_id']


    train = ensemble_data.iloc[:num_train, :]
    train[y_label[0]] = ensemble_train[y_label].values
    test = ensemble_data.iloc[num_train:, :]
    # for offline
    if not args.online:
        test[y_label[0]] = ensemble_test[y_label].values
    ensemble_train, ensemble_test = train, test
    print(ensemble_train.info())
    print(ensemble_train.head())
    print(ensemble_test.info())
    print(ensemble_test.head())

    if args.online:
        ALL_FEATURE_TRAIN_FILE = 'deep_feature_train' + '.' + fmt
        ALL_FEATURE_TEST_FILE = 'deep_feature_test' + '.' + fmt

    else:
        ALL_FEATURE_TRAIN_FILE = 'deep_feature_train' + str(kfold) + '.' + fmt
        ALL_FEATURE_TEST_FILE = 'deep_feature_test' + str(kfold) + '.' + fmt

    store_data(ensemble_train, os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)
    store_data(ensemble_test, os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)

    deep_col_feature_store_dir = os.path.join(col_feature_store_dir, "deep")
    if not os.path.exists(deep_col_feature_store_dir):
        os.mkdir(deep_col_feature_store_dir)

    tasks_args = []
    input_features = id_features + features_to_train
    for feature in set(input_features):
        train_file = feature + '_train' + '.' + fmt
        test_file = feature + '_test' + '.' + fmt
        param = (ensemble_train[[feature]], os.path.join(deep_col_feature_store_dir, train_file), fmt)
        tasks_args.append(param)
        param = (ensemble_test[[feature]], os.path.join(deep_col_feature_store_dir, test_file), fmt)
        tasks_args.append(param)


    if args.online:
        feature = y_label[0]
        train_file = feature + '_train' + '.' + fmt
        param = (ensemble_train[[feature]], os.path.join(deep_col_feature_store_dir, train_file), fmt)
        tasks_args.append(param)
    else:
        feature = y_label[0]
        train_file = feature + '_train' + '.' + fmt
        test_file = feature + '_test' + '.' + fmt
        param = (ensemble_test[[feature]], os.path.join(deep_col_feature_store_dir, test_file), fmt)
        tasks_args.append(param)
        param = (ensemble_train[[feature]], os.path.join(deep_col_feature_store_dir, train_file), fmt)
        tasks_args.append(param)


    def feature_saver(args):
        df, path, fmt = args
        res = store_data(df, path, fmt)
        return res


    start_time_1 = time.time()
    Executor = ProcessPoolExecutor
    with Executor(max_workers=num_workers) as executor:
        for file in executor.map(feature_saver, tasks_args):
            print('%s saved' % file)







