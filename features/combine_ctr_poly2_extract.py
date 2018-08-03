# coding:utf8
import os
import argparse
import sys

sys.path.append('..')
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count

import pandas as pd
import numpy as np
from common.utils import read_data, store_data
from conf.modelconf import get_data_file, data_dir, feature_dtype_map
from common.utils import count_combine_feat_ctr
from common import utils

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pool-type', help='pool type, threads or process, here use process for more performance')
parser.add_argument('-n', '--num-workers', help='workers num in pool', default=cpu_count())
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
parser.add_argument('-d', '--discretization', help='discrezatize or not some features', action='store_true')
parser.add_argument('-m', '--max-face-attr', help='attribute get max face used for each photo, options: appearance, scale, age, default=scale', default='scale')
args = parser.parse_args()




if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_face.txt',
                                                                                              args.online)
        TEST_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_face.txt',
                                                                                             args.online)
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction.txt')
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_interaction.txt')

        VISUAL_FEATURE_TRAIN_FILE = 'visual_feature_train' + '.' + fmt
        VISUAL_FEATURE_TEST_FILE = 'visual_feature_test' + '.' + fmt
        TEXT_FEATURE_FILE = 'text_feature' + '.' + fmt



    else:
        TRAIN_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_face' + str(kfold) + '.txt', args.online)
        TEST_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_face' + str(kfold) + '.txt', args.online)

        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'test_interaction' + str(kfold) + '.txt', online=False)

        VISUAL_FEATURE_TRAIN_FILE = 'visual_feature_train' + str(kfold) + '.' + fmt
        VISUAL_FEATURE_TEST_FILE = 'visual_feature_test' + str(kfold) + '.' + fmt
        TEXT_FEATURE_FILE = 'text_feature' + str(kfold) + '.' + fmt

    text_data = read_data(os.path.join(feature_store_dir, TEXT_FEATURE_FILE), fmt)
    print(text_data.info())

    face_train = pd.read_csv(TRAIN_FACE,
                             sep='\t',
                             header=None,
                             names=['photo_id', 'faces'])

    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                                  sep='\t',
                                  header=None,
                                  names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time',
                                         'duration_time'])

    face_test = pd.read_csv(TEST_FACE,
                            sep='\t',
                            header=None,
                            names=['photo_id', 'faces'])

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

    visual_train = read_data(os.path.join(feature_store_dir, VISUAL_FEATURE_TRAIN_FILE), fmt)
    visual_test = read_data(os.path.join(feature_store_dir, VISUAL_FEATURE_TEST_FILE), fmt)

    print(visual_train.head())
    print(visual_test.head())
    user_item_train = pd.merge(user_item_train, visual_train, how='left', on=['user_id', 'photo_id'])
    user_item_test = pd.merge(user_item_test, visual_test, how='left', on=['user_id', 'photo_id'])
    user_item_train = pd.merge(user_item_train, text_data, how='left', on=['photo_id'])
    user_item_test = pd.merge(user_item_test, text_data, how='left', on=['photo_id'])


    num_face_train = face_train.shape[0]
    face_data = pd.concat([face_train, face_test]).reset_index(drop=True)
    face_data['faces'] = face_data['faces'].apply(eval)
    face_data['face_num'] = face_data['faces'].apply(lambda l: len(l))
    face_data['man_num'] = face_data['faces'].apply(lambda lists: len([1 for l in lists if l[1] == 1]))
    face_data['woman_num'] = face_data['faces'].apply(lambda lists: len([1 for l in lists if l[1] == 0]))


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

    user_item_train = pd.merge(user_item_train, face_data.loc[:num_face_train], how='left', on=['photo_id'])
    user_item_test = pd.merge(user_item_test, face_data.loc[num_face_train:], how='left', on=['photo_id'])
    # face data is missing half
    for df in [user_item_train, user_item_test]:
        df['face_num'].fillna(0, inplace=True)
        df['woman_num'].fillna(0, inplace=True)
        df['man_num'].fillna(0, inplace=True)
        df['scale'].fillna(-1, inplace=True)
        df['gender'].fillna(-1, inplace=True)
        df['age'].fillna(-1, inplace=True)
        df['appearance'].fillna(-1, inplace=True)


    cate_cols = ['face_num', 'woman_num', 'man_num', 'gender', 'age', 'appearance', 'cover_length', 'duration_time', 'time', 'photo_cluster_label', 'text_cluster_label']
    if args.discretization:
        for col in cate_cols:
            func_name = col + '_discretization'
            func = getattr(utils, func_name) if hasattr(utils, func_name) else None
            if func is not None and callable(func):
                print(func.__name__)
                user_item_train[col] = user_item_train[col].apply(func)
                user_item_test[col] = user_item_test[col].apply(func)


    combine_cols = []
    for col1, col2 in itertools.permutations(cate_cols, 2):
        col = col1 + '_' + col2 + '_ctr'
        user_item_train[col], user_item_test[col] = count_combine_feat_ctr(
                                                                        user_item_train[col1].astype(str).values,
                                                                        user_item_train[col2].astype(str).values,
                                                                        user_item_test[col1].astype(str).values,
                                                                       user_item_test[col2].astype(str).values,
                                                                       user_item_train['click'].values)
        combine_cols.append(col)



    print(combine_cols)
    if args.online:
        COMBINE_TRAIN_FEATURE_FILE = 'combine_ctr2_feature_train' + '.' + fmt
        COMBINE_TEST_FEATURE_FILE = 'combine_ctr2_feature_test' + '.' + fmt

    else:
        COMBINE_TRAIN_FEATURE_FILE = 'combine_ctr2_feature_train' + str(kfold) + '.' + fmt
        COMBINE_TEST_FEATURE_FILE = 'combine_ctr2_feature_test' + str(kfold) + '.' + fmt

    combine_train = user_item_train[['user_id', 'photo_id'] + combine_cols]
    combine_test = user_item_test[['user_id', 'photo_id'] + combine_cols]
    combine_train.sort_values(['user_id', 'photo_id'], inplace=True)
    combine_test.sort_values(['user_id', 'photo_id'], inplace=True)
    print(combine_train.info())
    print(combine_test.info())

    print(np.sum(combine_train.isnull()))
    print(np.sum(combine_test.isnull()))

    if not os.path.exists(feature_store_dir):
        os.mkdir(feature_store_dir)
    store_data(combine_train, os.path.join(feature_store_dir, COMBINE_TRAIN_FEATURE_FILE), fmt)
    store_data(combine_test, os.path.join(feature_store_dir, COMBINE_TEST_FEATURE_FILE), fmt)

    # column
    tasks_args = []
    for col in set(combine_train.columns) - set(['user_id', 'photo_id']):
        tasks_args.append((combine_train[[col]], os.path.join(col_feature_store_dir, col + '_train.csv'), fmt))
        tasks_args.append((combine_test[[col]], os.path.join(col_feature_store_dir, col + '_test.csv'), fmt))


    def feature_saver(args):
        df, path, fmt = args
        res = store_data(df, path, fmt)
        return res


    start_time_1 = time.time()
    Executor = ThreadPoolExecutor if args.pool_type == 'thread' else ProcessPoolExecutor
    with Executor(max_workers=int(args.num_workers)) as executor:
        for file in executor.map(feature_saver, tasks_args):
            print('%s saved' % file)
    print ("%s pool execution in %s seconds" % (args.pool_type, str(time.time() - start_time_1)))
