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
    uint64_cols = ['user_id', 'photo_id', 'time']
    uint32_cols = ['playing_sum', 'browse_time_diff', 'duration_sum']
    uint16_cols = ['browse_num', 'exposure_num', 'click_num', 'duration_time', 'like_num', 'follow_num']
    uint8_cols = ['cover_length', 'man_num', 'woman_num', 'face_num']
    bool_cols = ['have_face_cate']
    float64_cols = ['clicked_ratio','non_face_click_favor', 'face_click_favor', 'man_favor', 'woman_avg_age', 'playing_freq', 'woman_age_favor', 'woman_yen_value_favor', 'human_scale', 'woman_favor', 'click_freq', 'woman_cv_favor', 'man_age_favor', 'man_yen_value_favor', 'follow_ratio', 'man_scale', 'browse_freq', 'man_avg_age', 'man_cv_favor', 'man_avg_attr', 'playing_ratio', 'woman_scale', 'click_ratio', 'human_avg_age', 'woman_avg_attr', 'like_ratio', 'cover_length_favor', 'human_avg_attr', 'avg_tfidf']
    ensemble_train[uint64_cols] = ensemble_train[uint64_cols].astype('uint64')
    ensemble_train[uint32_cols] = ensemble_train[uint32_cols].astype('uint32')
    ensemble_train[uint16_cols] = ensemble_train[uint16_cols].astype('uint16')
    ensemble_train[uint8_cols] = ensemble_train[uint8_cols].astype('uint8')
    ensemble_train[bool_cols] = ensemble_train[bool_cols].astype('bool')
    ensemble_train[float64_cols] = ensemble_train[float64_cols].astype('float32')
    ensemble_train[y_label] = ensemble_train[y_label].astype('bool')
    print(ensemble_train.info())
    
    ensemble_test[uint64_cols] = ensemble_test[uint64_cols].astype('uint64')
    ensemble_test[uint32_cols] = ensemble_test[uint32_cols].astype('uint32')
    ensemble_test[uint16_cols] = ensemble_test[uint16_cols].astype('uint16')
    ensemble_test[uint8_cols] = ensemble_test[uint8_cols].astype('uint8')
    ensemble_test[bool_cols] = ensemble_test[bool_cols].astype('bool')
    ensemble_test[float64_cols] = ensemble_test[float64_cols].astype('float32')
    print(ensemble_test.info())
    
    ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
    ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.' + fmt
    
    ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
    ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.' + fmt
    
    store_data(ensemble_train, os.path.join(feature_store_path, ALL_FEATURE_TRAIN_FILE), fmt)
    store_data(ensemble_test, os.path.join(feature_store_path, ALL_FEATURE_TEST_FILE), fmt)
