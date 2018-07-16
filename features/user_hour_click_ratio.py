#coding:utf8
import os
import argparse
import sys
sys.path.append("..")
    
import pandas as pd

from common.utils import read_data, store_data, FeatureMerger

from conf.modelconf import *
from multiprocessing import cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=1)
args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    n = cpu_count()
    if args.online:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_interaction.txt')
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_interaction.txt')
        PHOTO_FEATURE_FILE = 'photo_feature' + '.' + fmt

    else:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)

        PHOTO_FEATURE_FILE = 'photo_feature' + str(kfold) + '.' + fmt


    
    fm_trainer = FeatureMerger(col_feature_store_dir, ['time_cate', 'click', 'click_ratio'], fmt=fmt, data_type='train', pool_type='process', num_workers=n)
    fm_tester = FeatureMerger(col_feature_store_dir, ['time_cate'], fmt=fmt, data_type='test', pool_type='process', num_workers=n)
    
    user_item_train = fm_trainer.merge()
    print(user_item_train.info())
    user_item_test = fm_tester.merge()
    print(user_item_test.info())

    
    user_item_train['hour_click_ratio'] = user_item_train.set_index(['user_id', 'time_cate']).groupby(level=['user_id', 'time_cate'])['click'].transform('mean').values
    
    user_item_test = pd.merge(user_item_test, user_item_train[['user_id', 'time_cate', 'hour_click_ratio']].drop_duplicates(), how='left', on=['user_id', 'time_cate'])

    user_item_test['hour_click_ratio'].fillna(0, inplace=True)
    store_data(user_item_train[['user_id', 'photo_id', 'hour_click_ratio']], os.path.join(col_feature_store_dir, 'hour_click_ratio_train.csv'), fmt)
    
    store_data(user_item_test[['user_id', 'photo_id', 'hour_click_ratio']], os.path.join(col_feature_store_dir, 'hour_click_ratio_test.csv'), fmt)
    