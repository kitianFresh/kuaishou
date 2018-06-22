#coding:utf8
import os
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
sys.path.append("..")
    
import pandas as pd
import numpy as np

from common.utils import read_data, store_data, FeatureMerger
from conf.modelconf import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-p', '--pool-type', help='pool type, threads or process, here use process for more performance')
parser.add_argument('-n', '--num-workers', help='workers num in pool')
args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv' 
    pool_type = args.pool_type if args.pool_type else 'process'
    n = args.num_workers if args.num_workers else 8
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)
        
    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'
    
    feature_to_use = user_features + photo_features + time_features
    
    fm_trainer = FeatureMerger(col_feature_store_path, feature_to_use+y_label, fmt=fmt, data_type='train', pool_type=pool_type, num_workers=n)
    fm_tester = FeatureMerger(col_feature_store_path, feature_to_use, fmt=fmt, data_type='test', pool_type=pool_type, num_workers=n)
    
    ensemble_train = fm_trainer.merge()
    print(ensemble_train.info())
    ensemble_test = fm_tester.merge()
    print(ensemble_test.info())
    
    print(feature_to_use)

    print(ensemble_train.info())

    print(ensemble_test.info())
    
    ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
    ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.' + fmt
    
    ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
    ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.' + fmt
    
    store_data(ensemble_train, os.path.join(feature_store_path, ALL_FEATURE_TRAIN_FILE), fmt)
    store_data(ensemble_test, os.path.join(feature_store_path, ALL_FEATURE_TEST_FILE), fmt)
