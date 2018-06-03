#coding:utf8
import os
import argparse
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from common.utils import read_data, store_data

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
    return face_data


parser = argparse.ArgumentParser()
# parser.add_argument('-s', '--sample',type=str2bool, help='use sample data or full data', required=True)
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    
    TRAIN_FACE = '../sample/train_face.txt' if USE_SAMPLE else '../data/train_face.txt'
    TEST_FACE = '../sample/test_face.txt' if USE_SAMPLE else '../data/test_face.txt'
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
    
    face_data = pd.concat([face_train, face_test])
    face_tata = add_face_feature(face_data)
    face_data.drop(['faces'],axis=1,inplace=True)
    face_data.fillna(0, inplace=True)

    FACE_FEATURE_FILE = 'face_feature'
    FACE_FEATURE_FILE = FACE_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else FACE_FEATURE_FILE + '.' + fmt
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)    
    store_data(face_data, os.path.join(feature_store_path, FACE_FEATURE_FILE), fmt)
