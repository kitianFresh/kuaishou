# coding:utf8

import os
import argparse
import sys

sys.path.append("..")

import pandas as pd

from common.base import Feature, Table

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-p', '--pool-type', help='pool type, threads or process, here use process for more performance')
parser.add_argument('-n', '--num-workers', help='workers num in pool')
args = parser.parse_args()


class HourClickTableTrain(Table):
    _columns = ['time_cate', 'click', 'click_ratio']
    _table_type = 'train'


class HourClickTableTest(Table):
    _columns = ['time_cate']
    _table_type = 'test'


if __name__ == '__main__':

    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    pool_type = args.pool_type if args.pool_type else 'process'
    n = args.num_workers if args.num_workers else 8

    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)

    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'

    train_table = HourClickTableTrain('hour_click_num', features=None,
                 table_dir=feature_store_path,
                 merge_by_columns=True,
                 col_feature_dir=col_feature_store_path,num_workers=n)

    user_item_train = train_table.df
    # print(user_item_train.info())
    test_table = HourClickTableTest('hour_click_num', features=None,
                 table_dir=feature_store_path,
                 merge_by_columns=True,
                 col_feature_dir=col_feature_store_path,num_workers=n)
    user_item_test = test_table.df
    # print(user_item_test.info())


    user_item_train['hour_click_num'] = \
            user_item_train.set_index(['user_id', 'time_cate']).groupby(level=['user_id', 'time_cate'])[
                'click'].transform(
                'sum').values
    hour_click_ratio_feat_train = Feature('hour_click_num', feature_type='train', feature_dir=col_feature_store_path, fmt=fmt,
                           feature_data=user_item_train[['user_id', 'photo_id', 'hour_click_num']])

    hour_click_ratio_feat_train.save()

    user_item_test = pd.merge(user_item_test,
                              user_item_train[['user_id', 'time_cate', 'hour_click_num']].drop_duplicates(),
                              how='left', on=['user_id', 'time_cate'])

    user_item_test['hour_click_nums'].fillna(0, inplace=True)

    hour_click_ratio_feat_test = Feature('hour_click_num', feature_type='test', feature_dir=col_feature_store_path, fmt=fmt,
                           feature_data=user_item_train[['user_id', 'photo_id', 'hour_click_num']])

    hour_click_ratio_feat_test.save()
