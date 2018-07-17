#coding:utf8
import os
import argparse
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

from common.utils import read_data, store_data
from conf.modelconf import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=1)
args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_interaction.txt')
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_interaction.txt')

        FACE_FEATURE_FILE = 'face_feature' + '.' + fmt
        TEXT_FEATURE_FILE = 'text_feature' + '.' + fmt
    else:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)

        FACE_FEATURE_FILE = 'face_feature' + str(kfold) + '.' + fmt
        TEXT_FEATURE_FILE = 'text_feature' + str(kfold) + '.' + fmt

    face_data = read_data(os.path.join(feature_store_dir, FACE_FEATURE_FILE), fmt)
    print(face_data.info())
    text_data = read_data(os.path.join(feature_store_dir, TEXT_FEATURE_FILE), fmt)
    print(text_data.info())


    user_item_train = pd.read_csv(TRAIN_USER_INTERACT, 
                                 sep='\t', 
                                 header=None, 
                                 names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])

    user_item_test = pd.read_csv(TEST_USER_INTERACT,
                                 sep='\t', 
                                 header=None, 
                                 names=['user_id', 'photo_id', 'time', 'duration_time'])

    user_item_train = pd.merge(user_item_train, face_data,
                          how="left",
                          on=['photo_id'])

    user_item_train = pd.merge(user_item_train, text_data,
                          how="left",
                          on=['photo_id'])

    user_item_test = pd.merge(user_item_test, face_data,
                               how="left",
                               on=['photo_id'])

    user_item_test = pd.merge(user_item_test, text_data,
                               how="left",
                               on=['photo_id'])

    print(user_item_train.info())
    print(user_item_test.info())
    items = pd.DataFrame()
    common = ['text_cluster_label']
    items[common] = user_item_train[common]

    items['text_cluster_exposure_num'] = user_item_train['photo_id'].groupby(user_item_train['text_cluster_label']).transform('count')
    items['text_cluster_clicked_num'] = user_item_train['click'].groupby(user_item_train['text_cluster_label']).transform('sum')
    items['text_clicked_ratio'] = items['text_cluster_clicked_num'] / items['text_cluster_exposure_num']
    items.drop_duplicates(['text_cluster_label'], inplace=True)
    
    # 对物品店击率做贝叶斯平滑
    I, C = items['text_cluster_exposure_num'].values, items['text_cluster_clicked_num'].values
    #bs.update(I, C, 10000, 0.0000000001)
    #print(bs.alpha, bs.beta)
    #alpha_item, beta_item = bs.alpha, bs.beta
    alpha_item, beta_item = 2.8072236088257325, 13.280311727786964
    ctr = []
    for i in range(len(I)):
        ctr.append((C[i]+alpha_item)/(I[i]+alpha_item+beta_item))
    items['text_clicked_ratio'] = ctr
    items.drop(['text_cluster_exposure_num', 'text_cluster_clicked_num'], axis=1, inplace=True)


    user_item_data = pd.concat([user_item_train,
                                user_item_test])
    num_train, num_test = user_item_train.shape[0], user_item_test.shape[0]

    common = list(set(text_data.columns) | set(face_data.columns) - set(['user_id']))
    photo_data = user_item_data[common].copy()
    photo_data['exposure_num'] = user_item_data['photo_id'].groupby(user_item_data['photo_id']).transform('count')
    photo_data['text_cluster_exposure_num'] = user_item_data['photo_id'].groupby(user_item_data['text_cluster_label']).transform('count')
    photo_data.drop_duplicates(inplace=True)
    photo_data = pd.merge(photo_data, items, how='left', on=['text_cluster_label'])

    # photo_data.clicked_ratio.fillna(alpha_item/(alpha_item+beta_item), inplace=True)
    
    print(photo_data.info())
    print(np.sum(photo_data.isnull()))

    photo_data.fillna(-1, inplace=True)
    photo_data['have_face_cate'] = photo_data['face_num'].apply(lambda x: x >= 1).astype(feature_dtype_map['have_face_cate'])
    photo_data['have_text_cate'] = photo_data['cover_length'].apply(lambda x: x > 0).astype(feature_dtype_map['have_text_cate'])
    photo_data['text_class_label'] = photo_data['text_class_label'].astype(feature_dtype_map['text_class_label'])
    photo_data['text_cluster_label'] = photo_data['text_cluster_label'].astype(feature_dtype_map['text_cluster_label'])
    print(photo_data.info())
    
    if args.online:
        PHOTO_FEATURE_FILE = 'photo_feature' + '.' + fmt
    else:
        PHOTO_FEATURE_FILE = 'photo_feature' + str(kfold) + '.' + fmt

    if not os.path.exists(feature_store_dir):
        os.mkdir(feature_store_dir)
    store_data(photo_data, os.path.join(feature_store_dir, PHOTO_FEATURE_FILE), fmt)