#coding=utf-8

"""
@author: coffee
@time: 18-7-2 下午2:27
"""
import os
import argparse
import sys

sys.path.append('..')

from photo_cluster import load_data,sample_data_transform_store,data_transform_store,train_cluster_model,cluster_model_predict
import numpy as np
from common.utils import store_data
from conf.modelconf import *
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='use online data', action='store_true')
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
args = parser.parse_args()


if __name__ == '__main__':
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction.txt')
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction.txt')
    else:
        TRAIN_USER_INTERACT, offline_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, offline_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)

    TRAIN_VISUAL_MATRIX = os.path.join(online_data_dir, 'visual_train_matrix.blp')
    TEST_VISUAL_MATRIX = os.path.join(online_data_dir, 'visual_test_matrix.blp')
    TRAIN_VISUAL_PHOTO_ID = os.path.join(online_data_dir, 'visual_train_photo_id.blp')
    TEST_VISUAL_PHOTO_ID = os.path.join(online_data_dir, 'visual_test_photo_id.blp')
    VISUAL_CLUSTER_MODEL = os.path.join(online_data_dir, 'visual_cluster_model_kmeans.pkl')

    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                                  sep='\t',
                                  header=None,
                                  usecols =[0,1],
                                  names=['user_id', 'photo_id'])

    user_item_test = pd.read_csv(TEST_USER_INTERACT,
                                  sep='\t',
                                  header=None,
                                  usecols =[0,1],
                                  names=['user_id', 'photo_id'])


    def concat(X, Y, copy=False):
        """Return an array of references if copy=False"""
        if copy is True:  # deep copy
            return np.append(X, Y, axis=0)
        len_x, len_y = len(X), len(Y)
        ret = np.array([None for _ in range(len_x + len_y)])
        for i in range(len_x):
            ret[i] = X[i]
        for j in range(len_y):
            ret[len_x + j] = Y[j]
        return ret

    if os.path.exists(TRAIN_VISUAL_MATRIX) and os.path.exists(TEST_VISUAL_MATRIX):
        train_matrix,train_photo_id = load_data(TRAIN_VISUAL_MATRIX,TRAIN_VISUAL_PHOTO_ID)
        test_matrix,test_photo_id = load_data(TEST_VISUAL_MATRIX,TEST_VISUAL_PHOTO_ID)
    else:
        raise IOError('No matrix')

    cluster_nums = 50
    if os.path.exists(VISUAL_CLUSTER_MODEL):
        visual_feature_train = cluster_model_predict(VISUAL_CLUSTER_MODEL,train_matrix,train_photo_id)
        visual_feature_test = cluster_model_predict(VISUAL_CLUSTER_MODEL, test_matrix, test_photo_id)
    else:
        visual_feature_train = train_cluster_model(train_matrix,train_photo_id,VISUAL_CLUSTER_MODEL,cluster_nums)
        visual_feature_test = cluster_model_predict(VISUAL_CLUSTER_MODEL, test_matrix, test_photo_id)

    print(np.sum(visual_feature_train.isnull()))
    print(np.sum(visual_feature_test.isnull()))
    visual_feature_train = pd.merge(user_item_train,visual_feature_train,how='left',on=['photo_id'])
    visual_feature_test = pd.merge(user_item_test,visual_feature_test,how='left',on=['photo_id'])

    print(visual_feature_train.info())
    print(visual_feature_test.info())
    print(np.sum(visual_feature_train.isnull()))
    print(np.sum(visual_feature_test.isnull()))

    visual_feature_train.fillna(-1, inplace=True)
    visual_feature_test.fillna(-1, inplace=True)
    visual_feature_train['photo_cluster_label'] = visual_feature_train['photo_cluster_label'].astype(feature_dtype_map['photo_cluster_label'])
    visual_feature_test['photo_cluster_label'] = visual_feature_test['photo_cluster_label'].astype(feature_dtype_map['photo_cluster_label'])

    visual_feature_train.sort_values(['user_id', 'photo_id'], inplace=True)
    visual_feature_test.sort_values(['user_id', 'photo_id'], inplace=True)

    print(visual_feature_train.info())
    print(visual_feature_test.info())
    print(np.sum(visual_feature_train.isnull()))
    print(np.sum(visual_feature_test.isnull()))

    VISUAL_FEATURE_TRAIN_FILE = 'visual_feature_train' + '.' + fmt
    VISUAL_FEATURE_TEST_FILE = 'visual_feature_test' + '.' + fmt

    store_data(visual_feature_train, os.path.join(feature_store_dir, VISUAL_FEATURE_TRAIN_FILE), fmt)
    store_data(visual_feature_test, os.path.join(feature_store_dir, VISUAL_FEATURE_TEST_FILE), fmt)

    # column
    for col in set(visual_feature_train.columns) - set(['user_id', 'photo_id']):
        store_data(visual_feature_train[[col]], os.path.join(col_feature_store_dir, col + '_train.csv'), fmt)
        store_data(visual_feature_test[[col]], os.path.join(col_feature_store_dir, col + '_test.csv'), fmt)

