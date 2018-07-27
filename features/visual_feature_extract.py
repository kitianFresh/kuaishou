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
args = parser.parse_args()


if __name__ == '__main__':
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'

    if args.online:
        data_file_pre = 'online/'
    else:
        data_file_pre = 'offline/'

    TRAIN_USER_INTERACT = '../sample/'+data_file_pre+'train_interaction.txt' if USE_SAMPLE else '../data/'+data_file_pre+'train_interaction.txt'
    TEST_USER_INTERACT = '../sample/'+data_file_pre+'test_interaction.txt' if USE_SAMPLE else '../data/'+data_file_pre+'test_interaction.txt'

    TRAIN_VISUAL_DIR = '../sample/'+data_file_pre+'visual_train' if USE_SAMPLE else '../data/'+data_file_pre+'final_visual_train.zip'
    TEST_VISUAL_DIR = '../sample/'+data_file_pre+'visual_test' if USE_SAMPLE else '../data/'+data_file_pre+'final_visual_test.zip'

    TRAIN_VISUAL_MATRIX = '../sample/'+data_file_pre+'visual_train_matrix.blp' if USE_SAMPLE else '../data/'+data_file_pre+'visual_train_matrix.blp'
    TEST_VISUAL_MATRIX = '../sample/'+data_file_pre+'visual_test_matrix.blp' if USE_SAMPLE else '../data/'+data_file_pre+'visual_test_matrix.blp'


    TRAIN_VISUAL_PHOTO_ID = '../sample/'+data_file_pre+'visual_train_photo_id.blp' if USE_SAMPLE else '../data/'+data_file_pre+'visual_train_photo_id.blp'
    TEST_VISUAL_PHOTO_ID = '../sample/'+data_file_pre+'visual_test_photo_id.blp' if USE_SAMPLE else '../data/'+data_file_pre+'visual_test_photo_id.blp'

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

    user_item_data = pd.concat([user_item_train,user_item_test])


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

    matrix = None
    photo_id = None
    if os.path.exists(TRAIN_VISUAL_MATRIX) and os.path.exists(TEST_VISUAL_MATRIX):
        train_matrix,train_photo_id = load_data(TRAIN_VISUAL_MATRIX,TRAIN_VISUAL_PHOTO_ID)
        test_matrix,test_photo_id = load_data(TEST_VISUAL_MATRIX,TEST_VISUAL_PHOTO_ID)
        matrix = concat(train_matrix,test_matrix)
        photo_id = concat(train_photo_id,test_photo_id)
    else:
        if USE_SAMPLE:
            sample_data_transform_store(TRAIN_VISUAL_DIR,TRAIN_VISUAL_MATRIX,TRAIN_VISUAL_PHOTO_ID)
            sample_data_transform_store(TEST_VISUAL_DIR,TEST_VISUAL_MATRIX,TEST_VISUAL_PHOTO_ID)
            train_matrix,train_photo_id = load_data(TRAIN_VISUAL_MATRIX,TRAIN_VISUAL_PHOTO_ID)
            test_matrix, test_photo_id = load_data(TEST_VISUAL_MATRIX, TEST_VISUAL_PHOTO_ID)
            matrix = np.vstack([train_matrix, test_matrix])
            photo_id = np.append(train_photo_id, test_photo_id)
        else:
            data_transform_store(TRAIN_VISUAL_DIR, TRAIN_VISUAL_MATRIX, TRAIN_VISUAL_PHOTO_ID)
            data_transform_store(TEST_VISUAL_DIR, TEST_VISUAL_MATRIX, TEST_VISUAL_PHOTO_ID)
            train_matrix, train_photo_id = load_data(TRAIN_VISUAL_MATRIX, TRAIN_VISUAL_PHOTO_ID)
            test_matrix, test_photo_id = load_data(TEST_VISUAL_MATRIX, TEST_VISUAL_PHOTO_ID)
            matrix = np.vstack([train_matrix, test_matrix])
            photo_id = np.append(train_photo_id, test_photo_id)

    VISUAL_CLUSTER_MODEL = '../sample/'+data_file_pre+'visual_cluster_model' if USE_SAMPLE else '../data/'+data_file_pre+'visual_cluster_model'
    cluster_nums = 50
    visual_feature = None
    if os.path.exists(VISUAL_CLUSTER_MODEL + '_kmeans.pkl'):
        visual_feature = cluster_model_predict(VISUAL_CLUSTER_MODEL,matrix,photo_id)
    else:
        visual_feature = train_cluster_model(matrix,photo_id,VISUAL_CLUSTER_MODEL,cluster_nums)

    visual_feature = pd.merge(user_item_data,visual_feature,how='left',on=['photo_id'])
    visual_feature['photo_cluster_label'] = visual_feature['photo_cluster_label'].astype(feature_dtype_map['photo_cluster_label'])
    feature_store_path = '../sample/'+data_file_pre+'features' if USE_SAMPLE else '../data/'+data_file_pre+'features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)
    VISUAL_FEATURE_FILE = 'visual_feature'
    VISUAL_FEATURE_FILE = VISUAL_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else VISUAL_FEATURE_FILE + '.' + fmt
    store_data(visual_feature, os.path.join(feature_store_path, VISUAL_FEATURE_FILE), fmt)