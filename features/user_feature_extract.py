#coding:utf8
import os
import argparse
import sys
sys.path.append("..")

import pandas as pd
import numpy as np

from common.utils import read_data, store_data, BayesianSmoothing
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
        PHOTO_FEATURE_FILE = 'photo_feature' + '.' + fmt

    else:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)

        PHOTO_FEATURE_FILE = 'photo_feature' + str(kfold) + '.' + fmt


    photo_data = read_data(os.path.join(feature_store_dir, PHOTO_FEATURE_FILE), fmt)

    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                                 sep='\t', 
                                 header=None, 
                                 names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])

    user_item_test = pd.read_csv(TEST_USER_INTERACT,
                                 sep='\t', 
                                 header=None, 
                                 names=['user_id', 'photo_id', 'time', 'duration_time'])


    user_item_train = pd.merge(user_item_train, photo_data,
                              how='left',
                              on=['photo_id'])

    user_item_train.fillna(0, inplace=True)

    users = pd.DataFrame()
    users = pd.DataFrame()
    users['user_id'] = user_item_train['user_id']

    users['browse_num'] = user_item_train['user_id'].groupby(user_item_train['user_id']).transform('count')
    users['click_num'] = user_item_train['click'].groupby(user_item_train['user_id']).transform('sum')
    users['like_num'] = user_item_train['like'].groupby(user_item_train['user_id']).transform('sum')
    users['follow_num'] = user_item_train['follow'].groupby(user_item_train['user_id']).transform('sum')
    users['playing_sum'] = user_item_train['playing_time'].groupby(user_item_train['user_id']).transform('sum')
    users['duration_sum'] = user_item_train['duration_time'].groupby(user_item_train['user_id']).transform('sum')
    users['click_ratio'] = user_item_train['click'].groupby(user_item_train['user_id']).transform('mean')
    users['like_ratio'] = user_item_train['like'].groupby(user_item_train['user_id']).transform('mean')
    users['follow_ratio'] = user_item_train['follow'].groupby(user_item_train['user_id']).transform('mean')
    users['playing_ratio'] = users['playing_sum'] / users['duration_sum']
    
    def browse_time_diff(group):
        m1, m2 = group.min(), group.max()
        return (m2 - m1) / 1000
    users['browse_time_diff'] = user_item_train.groupby(['user_id'])['time'].transform(browse_time_diff)
    

    users.drop_duplicates(inplace=True)
    users.reset_index(drop=True, inplace=True)
    # 先验认为，点击频率越大，点击的可能性越高？试试
    users['click_freq'] = users['click_num'] / users['browse_time_diff']
    users['browse_freq'] = users['browse_num'] / users['browse_time_diff']
    users['playing_freq'] = users['playing_sum'] / users['browse_time_diff']
    
    # 用户点击视频中对人脸和颜值以及年龄的偏好，以后考虑离散化
    favor_cols = ['face_num', 'man_num', 'woman_num', 'man_scale', 'woman_scale', 'man_avg_age', 'woman_avg_age', 'man_avg_attr', 'woman_avg_attr', 'cover_length', 'playing_time', 'duration_time']
    favors = user_item_train.loc[user_item_train['click']==1, favor_cols+['user_id']]
    # 以下特征的统计，都可以看做和click_num 做过交叉
    favors['face_favor'] = favors['face_num'].groupby(favors['user_id']).transform('mean')
    favors['man_favor'] = favors['man_num'].groupby(favors['user_id']).transform('mean')
    favors['woman_favor'] = favors['woman_num'].groupby(favors['user_id']).transform('mean')
    favors['man_cv_favor'] = favors['man_scale'].groupby(favors['user_id']).transform('mean')
    favors['woman_cv_favor'] = favors['woman_scale'].groupby(favors['user_id']).transform('mean')
    favors['man_age_favor'] = favors['man_avg_age'].groupby(favors['user_id']).transform('mean')
    favors['woman_age_favor'] = favors['woman_avg_age'].groupby(favors['user_id']).transform('mean')
    favors['man_yen_value_favor'] = favors['man_avg_attr'].groupby(favors['user_id']).transform('mean')
    favors['woman_yen_value_favor'] = favors['woman_avg_attr'].groupby(favors['user_id']).transform('mean')
    favors['playing_favor'] = favors['playing_time'].groupby(favors['user_id']).transform('mean') # 点击数目和播放时间的交叉统计, 播放时间的偏爱
    favors['duration_favor'] = favors['duration_time'].groupby(favors['user_id']).transform('mean') # 点击数目和视频时长的交叉统计，对视频时长的偏爱，但是用户点击之前其实并不知道时长，这种特征不一定有效
    favors['playing_duration_ratio'] = favors['playing_time'] / favors['duration_time']
    favors['playing_duration_favor'] = favors['playing_duration_ratio'].groupby(favors['user_id']).transform('mean') # 视频播放时长、视频时长、click_num 三者交叉. 这个特征可以解释为
    
    
    def face_counts(group):
        have_faces = group > 0
        return 1. * np.sum(have_faces) / group.size

    def non_face_counts(group):
        have_faces = group == 0
        return 1. * np.sum(have_faces) / group.size
    favors['face_click_favor'] = favors['face_num'].groupby(favors['user_id']).transform(face_counts)
    favors['non_face_click_favor'] = 1 - favors['face_click_favor']
    
    # 文本平均长度作为偏爱率
    favors['cover_length_favor'] = favors['cover_length'].groupby(favors['user_id']).transform('mean')
    
    favors.drop_duplicates(['user_id'], inplace=True)
    favors.drop(favor_cols+['playing_duration_ratio'], axis=1, inplace=True)
    favors.reset_index(drop=True, inplace=True)
    
    users = pd.merge(users, favors,
                how='left',
                on=['user_id'])
    
    users.fillna(0, inplace=True)
    
    # 对用户点击率做贝叶斯平滑
    I, C = users['browse_num'].values, users['click_num'].values
    #bs.update(I, C, 10000, 0.0000000001)
    #print(bs.alpha, bs.beta)
    #alpha, beta = bs.alpha, bs.beta
    ctr = []
    for i in range(len(I)):
        ctr.append((C[i]+alpha)/(I[i]+alpha+beta))
    users['click_ratio'] = ctr
    
    print(users.info())
    def int2bin(x):
        res = []
        for i in '{0:016b}'.format(x):
            res.append(i)
        return '|'.join(res)
    users['temp_id'] = users['user_id'].apply(int2bin)
    ids = users['temp_id'].str.split('|', expand=True)  # 多名字分列
    users.pop('temp_id')
    ids.columns = ['uid0', 'uid1', 'uid2', 'uid3', 'uid4', 'uid5', 'uid6', 'uid7', 'uid8', 'uid9', 'uid10', 'uid11', 'uid12', 'uid13', 'uid14', 'uid15']
    ids[ids.columns] = ids[ids.columns].astype('uint8')
    users = users.join(ids)
    # users = pd.concat([users, ids], axis=1)
    if args.online:
        USER_FEATURE_FILE = 'user_feature' + '.' + fmt
    else:
        USER_FEATURE_FILE = 'user_feature' + str(kfold) + '.' + fmt


    if not os.path.exists(feature_store_dir):
        os.mkdir(feature_store_dir)
    store_data(users, os.path.join(feature_store_dir, USER_FEATURE_FILE), fmt)
