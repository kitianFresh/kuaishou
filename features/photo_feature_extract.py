#coding:utf8
import os
import argparse
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

from common.utils import read_data, store_data

def face_num_discretization(face_num):
    if face_num == 0:
        return 0
    elif face_num == 1:
        return 1
    elif face_num == 2:
        return 2
    elif face_num == 3:
        return 3
    else:
        return 4


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')

args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    
    FACE_FEATURE_FILE = 'face_feature'
    FACE_FEATURE_FILE = FACE_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else FACE_FEATURE_FILE + '.' + fmt
    face_data = read_data(FACE_FEATURE_FILE, fmt)

    TRAIN_USER_INTERACT = '../sample/train_interaction.txt' if USE_SAMPLE else '../data/train_interaction.txt'
    TEST_INTERACT = '../sample/test_interaction.txt' if USE_SAMPLE else '../data/test_interaction.txt'


    user_item_train = pd.read_csv(TRAIN_USER_INTERACT, 
                                 sep='\t', 
                                 header=None, 
                                 names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])
    user_item_train = user_item_train[['user_id', 'photo_id', 'time', 'duration_time']]

    user_item_test = pd.read_csv(TEST_INTERACT, 
                                 sep='\t', 
                                 header=None, 
                                 names=['user_id', 'photo_id', 'time', 'duration_time'])
    user_item_data = pd.concat([user_item_train, user_item_test])
    photo_data = pd.DataFrame()
    photo_data['photo_id'] = user_item_data['photo_id']
    photo_data['exposure_num'] = user_item_data['photo_id'].groupby(user_item_data['photo_id']).transform('count') 
    print(photo_data.info())
    photo_data.drop_duplicates(inplace=True)
    photo_data = pd.merge(photo_data, face_data,
                     how="left",
                     on=['photo_id'])

    photo_data.fillna(0, inplace=True)
    photo_data['face_num_class'] = photo_data['face_num'].apply(face_num_discretization)
    photo_data['have_face'] = photo_data['face_num'].apply(lambda x: x >= 1)
#     photo_data.drop(['face_num'], axis=1, inplace=True)
    
    PHOTO_FEATURE_FILE = 'photo_feature'
    PHOTO_FEATURE_FILE = PHOTO_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else PHOTO_FEATURE_FILE + '.' + fmt
    store_data(photo_data, PHOTO_FEATURE_FILE, fmt)