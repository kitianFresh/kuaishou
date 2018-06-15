#coding:utf8
import os
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
sys.path.append("..")
    
import pandas as pd
import numpy as np
from numpy.random import normal

from common.utils import read_data, store_data, FeatureMerger, BayesianSmoothing
from conf.modelconf import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-p', '--pool-type', help='pool type, threads or process, here use process for more performance')
parser.add_argument('-n', '--num-workers', help='workers num in pool')
parser.add_argument('-b', '--use-pretrained-beta', help='use pre trained beta alpha, beta', action="store_true")
args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv' 
    pool_type = args.pool_type if args.pool_type else 'process'
    n = args.num_workers if args.num_workers else 8
    use_pretrained = args.use_pretrained_beta
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)
        
    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'
    
    
    fm_trainer = FeatureMerger(col_feature_store_path, ['time_cate', 'click', 'click_ratio'], fmt=fmt, data_type='train', pool_type=pool_type, num_workers=n)
    fm_tester = FeatureMerger(col_feature_store_path, ['time_cate'], fmt=fmt, data_type='test', pool_type=pool_type, num_workers=n)
    
    user_item_train = fm_trainer.merge()
    print(user_item_train.info())
    user_item_test = fm_tester.merge()
    print(user_item_test.info())
    
    
    
    user_item_train['hour_click_ratio'] = user_item_train.set_index(['user_id', 'time_cate']).groupby(level=['user_id', 'time_cate'])['click'].transform('mean').values
    
    user_item_test = pd.merge(user_item_test, user_item_train[['user_id', 'time_cate', 'hour_click_ratio']].drop_duplicates(), how='left', on=['user_id', 'time_cate'])

    user_item_test['hour_click_ratio'].fillna(0, inplace=True)
    store_data(user_item_train[['user_id', 'photo_id', 'hour_click_ratio']], os.path.join(col_feature_store_path, 'hour_click_ratio_train.csv'), fmt)
    
    store_data(user_item_test[['user_id', 'photo_id', 'hour_click_ratio']], os.path.join(col_feature_store_path, 'hour_click_ratio_test.csv'), fmt)
    