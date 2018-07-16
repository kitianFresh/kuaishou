
# coding:utf8

import os
import argparse
import sys
import time
sys.path.append("../")
from multiprocessing import cpu_count

import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

from conf.modelconf import *
from common.utils import FeatureMerger, read_data, store_data
from common.base import Classifier


parser = argparse.ArgumentParser()
parser.add_argument('-a', '--all', help='use one ensemble table all, or merge by columns' ,action='store_true')
parser.add_argument('-n', '--num-workers', help='num used to merge columns', default=cpu_count())
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)


args = parser.parse_args()

if __name__ == '__main__':

    fmt = 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction.txt')
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + '.' + fmt
        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + '.' + fmt

    else:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction' + str(kfold) + '.txt', online=False)
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + str(kfold) + '.' + fmt
        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + str(kfold) + '.' + fmt

    ensemble_train = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)
    ensemble_test = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)

    print(ensemble_train.info())
    print(ensemble_test.info())

    if args.online:
        TRAIN_FILE = 'ensemble_feature_train' + '.libsvm'
        TEST_FILE = 'ensemble_feature_test' + '.libsvm'
    else:
        TRAIN_FILE = 'ensemble_feature_train' + str(kfold) + '.libsvm'
        TEST_FILE = 'ensemble_feature_test' + str(kfold) + '.libsvm'


    from sklearn.datasets import dump_svmlight_file
    X_train = ensemble_train[features_to_train].values
    y_train = ensemble_train[y_label].values
    X_test = ensemble_test[features_to_train].values
    y_test = ensemble_test[y_label].values
    dump_svmlight_file(X_train, y_train.ravel(), os.path.join(feature_store_dir, TRAIN_FILE))
    dump_svmlight_file(X_test, y_test.ravel(), os.path.join(feature_store_dir, TEST_FILE))
