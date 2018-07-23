# coding:utf8

import json
import argparse
import sys
sys.path.append("..")

import pandas as pd

from conf.modelconf import *
from multiprocessing import  cpu_count


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
parser.add_argument('-n', '--nums-split', help='nums to split to', default=cpu_count())

args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction.txt')
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction.txt')


    else:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)


    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                                  sep='\t',
                                  header=None,
                                  names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time',
                                         'duration_time'])
    if args.online:
        user_item_test = pd.read_csv(TEST_USER_INTERACT,
                                     sep='\t',
                                     header=None,
                                     names=['user_id', 'photo_id', 'time', 'duration_time'])
    else:
        user_item_test = pd.read_csv(TEST_USER_INTERACT,
                                     sep='\t',
                                     header=None,
                                     names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time',
                                            'duration_time'])

    n = args.nums_split
    photos_train = user_item_train['photo_id'].unique()
    photos_test = user_item_test['photo_id'].unique()
    num_train = user_item_train['photo_id'].nunique()
    num_test = user_item_test['photo_id'].nunique()

    interval_train, intervel_test = num_train // n, num_test // n
    data = {}
    for i in range(n):
        data['photo_ids_train' + str(i)] = photos_train[i*interval_train : (i+1)*interval_train]
        data['photo_ids_test' + str(i)] = photos_test[i*intervel_test : (i+1)*intervel_test]

    path = os.path.join(online_data_dir, 'photo_ids.json')
    with open(path, 'w') as f:
        json.dump(data, f)
    logging.info('Photo ids dumped in %s' % path)

