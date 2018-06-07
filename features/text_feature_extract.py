#coding:utf8
import os
import argparse
import sys
sys.path.append('..')

import numpy as np
import pandas as pd

from common.utils import read_data, store_data


parser = argparse.ArgumentParser()
# parser.add_argument('-s', '--sample',type=str2bool, help='use sample data or full data', required=True)
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    
    TRAIN_TEXT = '../sample/train_text.txt' if USE_SAMPLE else '../data/train_text.txt'
    TEST_TEXT = '../sample/test_text.txt' if USE_SAMPLE else '../data/test_text.txt'
    text_train = pd.read_csv(TRAIN_TEXT, 
                            sep='\t', 
                            header=None, 
                            names=['photo_id', 'cover_words'])

    print(text_train.info())

    text_test = pd.read_csv(TEST_TEXT, 
                            sep='\t', 
                            header=None, 
                            names=['photo_id', 'cover_words'])

    print(text_test.info())
    
    text_data = pd.concat([text_train, text_test])
    def words_to_list(words):
        if words == '0':
            return []
        else:
            return words.split(',')
        
    text_data['cover_words'] = text_data['cover_words'].apply(words_to_list)
    text_data['cover_length'] = text_data['cover_words'].apply(lambda words: len(words))
    text_data.drop(['cover_words'], axis=1, inplace=True)
    text_data.fillna(0, inplace=True)

    TEXT_FEATURE_FILE = 'text_feature'
    TEXT_FEATURE_FILE = TEXT_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else TEXT_FEATURE_FILE + '.' + fmt
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)
        
    store_data(text_data, os.path.join(feature_store_path, TEXT_FEATURE_FILE), fmt)
