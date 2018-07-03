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
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path) 
        
    
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
    path = os.path.join(feature_store_path,VISUAL_FEATURE_FILE)
    if os.path.exists(path):
        visual = read_data(os.path.join(feature_store_path,VISUAL_FEATURE_FILE),fmt)
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

    print(ensemble_train.info())
    
    ensemble_test = user_item_test[input_features]

    print(ensemble_test.info())
    
    
    ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
    ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.' + fmt
    
    ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
    ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.' + fmt
    
    store_data(ensemble_train, os.path.join(feature_store_path, ALL_FEATURE_TRAIN_FILE), fmt)
    store_data(ensemble_test, os.path.join(feature_store_path, ALL_FEATURE_TEST_FILE), fmt)
