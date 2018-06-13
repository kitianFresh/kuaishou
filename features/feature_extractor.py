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
    
    feature_to_use = ['exposure_num', 'clicked_num']
    
    fm_trainer = FeatureMerger(col_feature_store_path, ['exposure_num', 'clicked_num'], fmt=fmt, data_type='train', pool_type=pool_type, num_workers=n)
    fm_tester = FeatureMerger(col_feature_store_path, ['exposure_num'], fmt=fmt, data_type='test', pool_type=pool_type, num_workers=n)
    
    photo_train = fm_trainer.merge()
    print(photo_train.info())
    photo_test = fm_tester.merge()
    print(photo_test.info())
    
    items = photo_train[['photo_id', 'exposure_num', 'clicked_num']]
    items = items.drop_duplicates(['photo_id'])
    
    
    I, C = items['exposure_num'].values, items['clicked_num'].values
    if use_pretrained:
        alpha_item = 2.5171267342473382 if USE_SAMPLE else 2.8072236088257325
        beta_item = 7.087836849232511 if USE_SAMPLE else 13.280311727786964
    else:
        bs = BayesianSmoothing(1, 1)
        bs.update(I, C, 10000, 0.0000000001)
        print(bs.alpha, bs.beta)
        alpha_item, beta_item = bs.alpha, bs.beta
    ctr = []
    for i in range(len(I)):
        ctr.append((C[i]+alpha_item)/(I[i]+alpha_item+beta_item))
    items['clicked_ratio'] = ctr
    items.drop(['exposure_num', 'clicked_num'], axis=1, inplace=True)
    print(items.head(20))
    clicked_ratio_col_train = pd.merge(photo_train[['user_id', 'photo_id']], items[['photo_id', 'clicked_ratio']],
                                  how='left',
                                  on=['photo_id'])
    print(clicked_ratio_col_train.head())
    store_data(clicked_ratio_col_train, os.path.join(col_feature_store_path, 'clicked_ratio_train.csv'), fmt)
    
    sigma = items.clicked_ratio.std()
    u = alpha_item/(alpha_item+beta_item)
    items1 = photo_test[['photo_id', 'exposure_num']]
    print(photo_test.head())
    items1 = items1.drop_duplicates(['photo_id'])
    items1['exposure_num_sigma'] = items1['exposure_num'].apply(lambda x: np.exp(-x) * sigma)
    print(items1.head(20))
    items1['noise'] = items1['exposure_num_sigma'].apply(lambda x: normal(0, x))
    print(items1.head(20))
    items1['clicked_ratio'] = u + items1['noise']
    print(items1.head(20))
    items1.loc[items1['clicked_ratio']<0, ['clicked_ratio']] = 0
    print(items1.head(20))
    
    clicked_ratio_col_test = pd.merge(photo_test[['user_id', 'photo_id']], items1[['photo_id', 'clicked_ratio']],
                                  how='left',
                                  on=['photo_id'])
    print(clicked_ratio_col_test.head())

    store_data(clicked_ratio_col_test, os.path.join(col_feature_store_path, 'clicked_ratio_test.csv'), fmt)
    