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

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
args = parser.parse_args()

if __name__ == '__main__':
    USE_SAMPLE = args.sample
    TRAIN_TEXT = '../sample/train_text.txt' if USE_SAMPLE else '../data/train_text.txt'
    TEST_TEXT = '../sample/test_text.txt' if USE_SAMPLE else '../data/test_text.txt'

    TRAIN_USER_INTERACT = '../sample/train_interaction.txt' if USE_SAMPLE else '../data/train_interaction.txt'
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

    def classify_data_generate(click,words):
        if click == 1 and words != '':
            return words + '\t__label__' + 'click'
        elif click == 0 and words != '':
            return words + '\t__label__' + 'unclick'
        else:
            return ''



    CLASSIFY_TRAIN_DATA_PATH = '../sample/classify_train_data' if USE_SAMPLE else '../data/classify_train_data'
    CLASSIFY_TEST_DATA_PATH = '../sample/classify_test_data' if USE_SAMPLE else '../data/classify_test_data'

    if not os.path.exists(CLASSIFY_TRAIN_DATA_PATH) and not  os.path.exists(CLASSIFY_TEST_DATA_PATH):
        text_data_click['classify_text'] = map(lambda x, y: classify_data_generate(x, y), text_data_click['click'],
                                               text_data_click['cover_words'])
        text_data_click = text_data_click.sample(frac=1)
        size = text_data_click.shape[0]
        num = int(size * 0.8)
        train_data = text_data_click[:num]
        test_data = text_data_click[num:]
        classify_train_data_file = open(CLASSIFY_TRAIN_DATA_PATH, 'w')
        classift_test_data_file = open(CLASSIFY_TEST_DATA_PATH, 'w')
        for sentence in train_data['classify_text']:
            if sentence != '':
                classify_train_data_file.write(sentence)
                classify_train_data_file.write('\n')
        for sentence in test_data['classify_text']:
            if sentence != '':
                classift_test_data_file.write(sentence)
                classift_test_data_file.write('\n')

    CLASSIFY_MODEL_PATH = '../sample/classify_model' if USE_SAMPLE else '../data/classify_model'

    if not os.path.exists(CLASSIFY_MODEL_PATH + '.bin'):
        classifier = fasttext.supervised(CLASSIFY_TRAIN_DATA_PATH, CLASSIFY_MODEL_PATH, label_prefix='__label__')

    else:
        classifier = fasttext.load_model(CLASSIFY_MODEL_PATH + '.bin', label_prefix='__label__')
        result = classifier.test(CLASSIFY_TEST_DATA_PATH)
        print(result.precision)
        print(result.recall)
