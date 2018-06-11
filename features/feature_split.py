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
parser.add_argument('-p', '--pool-type', help='pool type, threads or process')
parser.add_argument('-n', '--num-workers', help='workers num in pool')
args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv' 
    pool_type = args.pool_type if args.pool_type else 'thread'
    n = args.num_workers if args.num_workers else 4
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
    
    user_item_train = pd.merge(user_item_train, users,
                          how='inner',
                          on=['user_id'])
    
    user_item_train = pd.merge(user_item_train, photo_data,
                              how='left',
                              on=['photo_id'])
    
    user_item_test = pd.merge(user_item_test, users,
                             how='inner',
                             on=['user_id'])
    
    user_item_test = pd.merge(user_item_test, photo_data,
                          how='left',
                          on=['photo_id'])
    

    
    print(user_item_train.columns)
    user_item_train.fillna(0, inplace=True)
    user_item_test.fillna(0, inplace=True)
    input_features = id_features + user_features + photo_features + time_features
    
    print(input_features)
    ensemble_train = user_item_train[input_features + y_label]
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
    
    ensemble_test = user_item_test[input_features]
    ensemble_test[uint64_cols] = ensemble_test[uint64_cols].astype('uint64')
    ensemble_test[uint32_cols] = ensemble_test[uint32_cols].astype('uint32')
    ensemble_test[uint16_cols] = ensemble_test[uint16_cols].astype('uint16')
    ensemble_test[uint8_cols] = ensemble_test[uint8_cols].astype('uint8')
    ensemble_test[bool_cols] = ensemble_test[bool_cols].astype('bool')
    ensemble_test[float64_cols] = ensemble_test[float64_cols].astype('float32')
    print(ensemble_test.info())
#     fmt = 'h5'
    tasks_args = []
    for feature in set(input_features) - set(id_features):
        train_file = feature  + '_train_sample' + '.' + fmt if USE_SAMPLE else feature + '_train' + '.' + fmt
        test_file = feature  + '_test_sample' + '.' + fmt if USE_SAMPLE else feature + '_test' + '.' + fmt
        args = (ensemble_train[id_features+[feature]], os.path.join(col_feature_store_path, train_file), fmt)
        tasks_args.append(args)
        args = (ensemble_test[id_features+[feature]], os.path.join(col_feature_store_path, test_file), fmt)
        tasks_args.append(args)
        
    feature = y_label[0]
    train_file = feature  + '_train_sample' + '.' + fmt if USE_SAMPLE else feature + '_train' + '.' + fmt
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
    print ("Thread pool execution in " + str(time.clock() - start_time_1), "seconds")
