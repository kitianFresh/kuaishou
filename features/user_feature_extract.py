#coding:utf8
import os
import argparse
import sys
sys.path.append("..")

import pandas as pd
import numpy as np

from common.utils import read_data, store_data

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')

args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)
        
    PHOTO_FEATURE_FILE = 'photo_feature'
    PHOTO_FEATURE_FILE = PHOTO_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else PHOTO_FEATURE_FILE + '.' + fmt
    photo_data = read_data(os.path.join(feature_store_path, PHOTO_FEATURE_FILE), fmt)

    TRAIN_USER_INTERACT = '../sample/train_interaction.txt' if USE_SAMPLE else '../data/train_interaction.txt'
    TEST_INTERACT = '../sample/test_interaction.txt' if USE_SAMPLE else '../data/test_interaction.txt'
    

    user_item_train = pd.read_csv(TRAIN_USER_INTERACT, 
                                 sep='\t', 
                                 header=None, 
                                 names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])

    user_item_test = pd.read_csv(TEST_INTERACT, 
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
    users['playing_ratio'] = (users['playing_sum'] / users['duration_sum'])
    
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
    users['face_favor'] = user_item_train.loc[user_item_train['click']==1, ['user_id', 'face_num']].groupby(user_item_train['user_id']).transform('mean').drop_duplicates(['user_id'])['face_num'].values
    users['man_favor'] = user_item_train.loc[user_item_train['click']==1, ['user_id', 'man_num']].groupby(user_item_train['user_id']).transform('mean').drop_duplicates(['user_id'])['man_num'].values
    users['woman_favor'] = user_item_train.loc[user_item_train['click']==1, ['user_id', 'woman_num']].groupby(user_item_train['user_id']).transform('mean').drop_duplicates(['user_id'])['woman_num'].values
    users['man_cv_favor'] = user_item_train.loc[user_item_train['click']==1, ['user_id', 'man_scale']].groupby(user_item_train['user_id']).transform('mean').drop_duplicates(['user_id'])['man_scale'].values
    users['woman_cv_favor'] = user_item_train.loc[user_item_train['click']==1, ['user_id', 'woman_scale']].groupby(user_item_train['user_id']).transform('mean').drop_duplicates(['user_id'])['woman_scale'].values
    users['man_age_favor'] = user_item_train.loc[user_item_train['click']==1, ['user_id', 'man_avg_age']].groupby(user_item_train['user_id']).transform('mean').drop_duplicates(['user_id'])['man_avg_age'].values
    users['woman_age_favor'] = user_item_train.loc[user_item_train['click']==1, ['user_id', 'woman_avg_age']].groupby(user_item_train['user_id']).transform('mean').drop_duplicates(['user_id'])['woman_avg_age'].values
    users['man_yen_value_favor'] = user_item_train.loc[user_item_train['click']==1, ['user_id', 'man_avg_attr']].groupby(user_item_train['user_id']).transform('mean').drop_duplicates(['user_id'])['man_avg_attr'].values
    users['woman_yen_value_favor'] = user_item_train.loc[user_item_train['click']==1, ['user_id', 'woman_avg_attr']].groupby(user_item_train['user_id']).transform('mean').drop_duplicates(['user_id'])['woman_avg_attr'].values

    USER_FEATURE_FILE = 'user_feature'
    USER_FEATURE_FILE = USER_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else USER_FEATURE_FILE + '.' + fmt
    
    
    store_data(users, os.path.join(feature_store_path, USER_FEATURE_FILE), fmt)

