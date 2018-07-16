#coding=utf-8

"""
@author: coffee
@time: 18-6-22 下午6:55
"""
import os
import argparse
import sys
import pandas as pd
import fasttext
sys.path.append('..')
from conf.modelconf import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold',default=0)
args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_text.txt',
                                                                                              args.online)
        TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_text.txt',
                                                                                             args.online)
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction.txt')

    else:
        TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_text' + str(kfold) + '.txt', online=False)
        TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_text' + str(kfold) + '.txt', online=False)

        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)

    # trained once, use anywhere, shared for all online or offline folds
    CLASSIFY_MODEL_PATH = os.path.join(data_dir, 'classify_model')
    CLASSIFY_TRAIN_DATA_PATH = os.path.join(data_dir, 'classify_train_data')
    CLASSIFY_TEST_DATA_PATH = os.path.join(data_dir, 'classify_test_data')

    user_interact_train = pd.read_csv(TRAIN_USER_INTERACT,
                                      sep='\t',
                                      usecols=[1,2],
                                      header=None,
                                      names=['photo_id','click'])

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
            return ''
        else:
            return words.replace(',',' ')


    text_data['cover_words'] = text_data['cover_words'].apply(words_to_list)

    text_data.fillna(0, inplace=True)

    text_data_click = pd.merge(user_interact_train,text_data,how='left',on=['photo_id'])

    #print(text_data_click.head())

    def classify_data_generate(click,words):
        if click == 1 and words != '':
            return words + '\t__label__' + 'click'
        elif click == 0 and words != '':
            return words + '\t__label__' + 'unclick'
        else:
            return ''


    if not os.path.exists(CLASSIFY_TRAIN_DATA_PATH) and not os.path.exists(CLASSIFY_TEST_DATA_PATH):
        text_data_click['classify_text'] = pd.Series(map(lambda x, y: classify_data_generate(x, y), text_data_click['click'],
                                                         text_data_click['cover_words']))
        #print(text_data_click['classify_text'])
        text_data_click = text_data_click.sample(frac=1)
        size = text_data_click.shape[0]

        num = int(size * 0.8)
        train_data = text_data_click[:num]
        test_data = text_data_click[num:]

        classify_train_data_file = open(CLASSIFY_TRAIN_DATA_PATH, 'w')
        classift_test_data_file = open(CLASSIFY_TEST_DATA_PATH, 'w')

        #print(train_data.head())
        for sentence in train_data['classify_text']:
            #print(sentence)
            if sentence != '':
                classify_train_data_file.write(sentence)
                classify_train_data_file.write('\n')
        for sentence in test_data['classify_text']:
            if sentence != '':
                classift_test_data_file.write(sentence)
                classift_test_data_file.write('\n')
        classify_train_data_file.close()
        classift_test_data_file.close()

    if not os.path.exists(CLASSIFY_MODEL_PATH + '.bin'):
        classifier = fasttext.supervised(CLASSIFY_TRAIN_DATA_PATH, CLASSIFY_MODEL_PATH, label_prefix='__label__')

    else:
        classifier = fasttext.load_model(CLASSIFY_MODEL_PATH + '.bin', label_prefix='__label__')
        result = classifier.test(CLASSIFY_TEST_DATA_PATH)
        print(result.precision)
        print(result.recall)
