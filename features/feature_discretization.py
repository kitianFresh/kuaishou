#coding:utf8
import os
import argparse
import sys
sys.path.append("..")
    
import pandas as pd
import numpy as np
from common import utils
from common.utils import *
from conf.modelconf import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
args = parser.parse_args()

if __name__ == '__main__':
    
    path = '../features'
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    
    ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
    ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.' + fmt
    ensemble_train = read_data(os.path.join(path, ALL_FEATURE_TRAIN_FILE), fmt)

    ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
    ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.' + fmt
    ensemble_test = read_data(os.path.join(path, ALL_FEATURE_TEST_FILE), fmt)

    print(ensemble_train.info())
    print(ensemble_test.info())

    all_features = list(ensemble_train.columns.values)
    print("all original features")
    print(all_features)
    y = ensemble_train[y_label].values
    
    ensemble_train = ensemble_train[id_features + features_to_train]
    ensemble_test = ensemble_test[id_features + features_to_train]
    num_train, num_test = ensemble_train.shape[0], ensemble_test.shape[0]
    ensemble_data = pd.concat([ensemble_train, ensemble_test])
    
    
    category_features = time_features + user_features + photo_features
    user_item_cate = pd.DataFrame()
    cates = []
    for col in category_features:
        name = col+'_discretization'
        func = getattr(utils, name) if hasattr(utils, name) else None
        if func is not None and callable(func):
            print(func.__name__)
            user_item_cate[col+'_cate'] = ensemble_data[col].apply(func)
            cates.append(col+'_cate')
        else:
            user_item_cate[col] = ensemble_data[col]
            
    print(user_item_cate.info())
    print(user_item_cate.head())
    user_item_cate[cates] = user_item_cate[cates].astype('uint8')
    user_item_cate = pd.concat([ensemble_data[id_features], user_item_cate], axis=1)
    
    cate_train = user_item_cate.iloc[:num_train, :]
    cate_train[y_label[0]] = y
    cate_test = user_item_cate.iloc[num_train:, :]
    print(cate_train.info())
    print(cate_train.head())
    print(cate_test.info())
    print(cate_test.head())
    
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)    
    store_data(face_data, os.path.join(feature_store_path, FACE_FEATURE_FILE), fmt)

    CATE_TRAIN_FILE = 'ensemble_cate_feature_train'
    CATE_TRAIN_FILE = CATE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else CATE_TRAIN_FILE + '.' + fmt
    store_data(cate_train, os.path.join(feature_store_path, CATE_TRAIN_FILE), fmt)

    CATE_TEST_FILE = 'ensemble_cate_feature_test'
    CATE_TEST_FILE = CATE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else CATE_TEST_FILE + '.' + fmt
    store_data(cate_test, os.path.join(feature_store_path, CATE_TEST_FILE), fmt)
