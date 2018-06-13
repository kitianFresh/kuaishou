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
    
    feature_to_use = ['exposure_num', 'clicked_num']
    
    fm_trainer = FeatureMerger(col_feature_store_path, ['exposure_num', 'clicked_num', 'clicked_ratio'], fmt=fmt, data_type='train', pool_type=pool_type, num_workers=n)
    fm_tester = FeatureMerger(col_feature_store_path, ['exposure_num'], fmt=fmt, data_type='test', pool_type=pool_type, num_workers=n)
    
    photo_train = fm_trainer.merge()
    print(photo_train.info())
    photo_test = fm_tester.merge()
    print(photo_test.info())
    
    items = photo_train['photo_id', 'exposure_num', 'clicked_num', 'clicked_ratio']
    items = photo_train.drop_duplicates(['photo_id'])
    I, C = items['exposure_num'].values, items['clicked_num'].values
    bs.update(I, C, 10000, 0.0000000001)
    print(bs.alpha, bs.beta)
    alpha_item, beta_item = bs.alpha, bs.beta
#     alpha_item, beta_item = 2.8072236088257325, 13.280311727786964
    ctr = []
    for i in range(len(I)):
        ctr.append((C[i]+alpha_item)/(I[i]+alpha_item+beta_item))
    items['clicked_ratio'] = ctr
    items.drop(['exposure_num', 'clicked_num'], axis=1, inplace=True)
    
    clicked_ratio_col_train = pd.merge(photo_train[['user_id', 'photo_id']], items[['photo_id', 'clicked_ratio']],
                                  how='left',
                                  on=['photo_id'])
    

    sigma = items.clicked_ratio.std()
    u = alpha_item/(alpha_item+beta_item)
    items = photo_test['photo_id', 'exposure_num']
    items = photo_train.drop_duplicates(['photo_id'])
    items['exposure_num_sigma'] = items['exposure_num'].apply(lambda x: np.exp(-x) * sigma)
    
    items['noise'] = items['exposure_num_sigma'].apply(lambda x: normal(0, x))
    items['clicked_ratio'] = u + items['noise']
    items.loc[items['clicked_ratio']<0, ['clicked_ratio']] = 0
    
    clicked_ratio_col_test = pd.merge(photo_test[['user_id', 'photo_id']], items[['photo_id', 'clicked_ratio']],
                                  how='left',
                                  on=['photo_id'])
    