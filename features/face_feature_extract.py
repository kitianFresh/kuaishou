#coding:utf8
import os
import argparse
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from common.utils import read_data, store_data
from conf.modelconf import *

def add_face_feature(face_data):
    face_data['faces'] = face_data['faces'].apply(eval)
    face_data['face_num'] = face_data['faces'].apply(lambda l : len(l))
    face_data['man_num'] = face_data['faces'].apply(lambda lists: len([1 for l in lists if l[1] == 1]))
    face_data['woman_num'] = face_data['faces'].apply(lambda lists: len([1 for l in lists if l[1] == 0]))
    face_data['man_scale'] = face_data['faces'].apply(lambda lists: sum([l[0] for l in lists if l[1] == 1]))
    face_data['woman_scale'] = face_data['faces'].apply(lambda lists: sum([l[0] for l in lists if l[1] == 0]))
    face_data['human_scale'] = face_data['man_scale'] + face_data['woman_scale']
    face_data['man_avg_age'] = face_data['faces'].apply(lambda lists: np.mean([l[2] for l in lists if l[1] == 1]))
    face_data['woman_avg_age'] = face_data['faces'].apply(lambda lists: np.mean([l[2] for l in lists if l[1] == 0]))
    face_data['human_avg_age'] = face_data['faces'].apply(lambda lists: np.mean([l[2] for l in lists]))
    face_data['man_avg_attr'] = face_data['faces'].apply(lambda lists: np.mean([l[3] for l in lists if l[1] == 1]))
    face_data['woman_avg_attr'] = face_data['faces'].apply(lambda lists: np.mean([l[3] for l in lists if l[1] == 0]))
    face_data['human_avg_attr'] = face_data['faces'].apply(lambda lists: np.mean([l[3] for l in lists]))
    face_data['man_num_ratio'] = 1. * face_data['man_num'] / face_data['face_num']
    face_data['woman_num_ratio'] = 1. * face_data['woman_num'] / face_data['face_num']
    return face_data


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=1)
args = parser.parse_args()

if __name__ == '__main__':
    
    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_face.txt', args.online)
        TEST_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_face.txt', args.online)
    else:
        TRAIN_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_face' + str(kfold) + '.txt', args.online)
        TEST_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_face' + str(kfold) + '.txt', args.online)

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
    face_data = add_face_feature(face_data)
    face_data.drop(['faces'],axis=1,inplace=True)
    face_data.fillna(0, inplace=True)

    if args.online:
        FACE_FEATURE_FILE = 'face_feature' + '.' + fmt
    else:
        FACE_FEATURE_FILE = 'face_feature' + str(kfold) + '.' + fmt

    if not os.path.exists(feature_store_dir):
        os.mkdir(feature_store_dir)
    store_data(face_data, os.path.join(feature_store_dir, FACE_FEATURE_FILE), fmt)
