# coding:utf8
import os
import argparse
import sys

sys.path.append('..')

import numpy as np
import pandas as pd

from common.utils import read_data, store_data
from conf.modelconf import *
from common import utils


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
parser.add_argument('-d', '--discretization', help='discrezatize or not some features', action='store_true')
parser.add_argument('-m', '--max-face-attr', help='attribute get max face used for each photo, options: appearance, scale, age', default='scale')
args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_face.txt',
                                                                                              args.online)
        TEST_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_face.txt',
                                                                                             args.online)
    else:
        TRAIN_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_face' + str(kfold) + '.txt', args.online)
        TEST_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_face' + str(kfold) + '.txt', args.online)

    print(TRAIN_FACE)
    print(TEST_FACE)
    face_train = pd.read_csv(TRAIN_FACE,
                             sep='\t',
                             header=None,
                             names=['photo_id', 'faces'])

    print(face_train.info())

    face_test = pd.read_csv(TEST_FACE,
                            sep='\t',
                            header=None,
                            names=['photo_id', 'faces'])

    print(face_test.info())

    face_data = pd.concat([face_train, face_test]).reset_index(drop=True)
    face_data['faces'] = face_data['faces'].apply(eval)


    # 0: scale, 1: gender, 2: age, 3: appearance
    def max_scale_face(faces):
        return '|'.join(map(str, faces[np.argmax(map(lambda x: x[0], faces))]))


    def max_appearance_face(faces):
        return '|'.join(map(str, faces[np.argmax(map(lambda x: x[3], faces))]))


    def max_age_face(faces):
        return '|'.join(map(str, faces[np.argmax(map(lambda x: x[2], faces))]))


    cols_dtype = {
        'scale': np.float32,
        'gender': np.int8,
        'age': np.int8,
        'appearance': np.float32,
    }

    # max_scale_face, max_age_face
    prefix = 'max_' + args.max_face_attr + '_'
    temp_col = prefix + 'face'
    if temp_col == 'max_scale_face':
        face_data['max_scale_face'] = face_data['faces'].apply(max_scale_face)
    elif temp_col == 'max_apperance_face':
        face_data[temp_col] = face_data['faces'].apply(max_appearance_face)
    elif temp_col == 'max_age_face':
        face_data['max_age_face'] = face_data['faces'].apply(max_age_face)

    faces = face_data[temp_col].str.split('|', expand=True)  # 多名字分列
    face_data.pop(temp_col)
    faces.columns = ['scale', 'gender', 'age', 'appearance']
    face_data = face_data.join(faces)

    for feat, dtype in cols_dtype.items():
        face_data[feat] = face_data[feat].astype(cols_dtype[feat])

    face_data.drop(['faces'], axis=1, inplace=True)

    if args.discretization:
        cate_cols = ['appearance', 'scale']
        for col in cate_cols:
            func_name = col + '_discretization'
            func = getattr(utils, func_name) if hasattr(utils, func_name) else None
            if func is not None and callable(func):
                print(func.__name__)
                face_data[col] = face_data[col].apply(func)

    if args.online:
        FACE_MAX_FEATURE_FILE = 'face_max_feature' + '.' + fmt
    else:
        FACE_MAX_FEATURE_FILE = 'face_max_feature' + str(kfold) + '.' + fmt

    if not os.path.exists(feature_store_dir):
        os.mkdir(feature_store_dir)
    store_data(face_data[['photo_id', 'scale', 'gender', 'age', 'appearance']], os.path.join(feature_store_dir, FACE_MAX_FEATURE_FILE), fmt)
