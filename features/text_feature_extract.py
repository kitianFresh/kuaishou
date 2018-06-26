# coding:utf8
import os
import argparse
import sys

sys.path.append('..')

import pandas as pd
import numpy as np
from gensim import corpora, models
from collections import Counter
import fasttext
from common.utils import read_data, store_data
from text_cluster import load_cluster_model,train_cluster_model,cluster_model_predict
from word_embedding import load_model,train_word2vec

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
args = parser.parse_args()

if __name__ == '__main__':

    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'

    TRAIN_TEXT = '../sample/train_text.txt' if USE_SAMPLE else '../data/train_text.txt'
    TEST_TEXT = '../sample/test_text.txt' if USE_SAMPLE else '../data/test_text.txt'

    TRAIN_USER_INTERACT = '../sample/train_interaction.txt' if USE_SAMPLE else '../data/train_interaction.txt'
    user_interact_train = pd.read_csv(TRAIN_USER_INTERACT,
                                      sep='\t',
                                      usecols=[1,2],
                                      header=None,
                                      names=['photo_id','click'])

    user_interact_train = user_interact_train[user_interact_train['click'] == 1]
    user_interact_train.drop(['click'],axis=1,inplace=True)

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


    text_data['cover_words_4_predict'] = text_data['cover_words']
    text_data['cover_words'] = text_data['cover_words'].apply(words_to_list)
    text_data['cover_length'] = text_data['cover_words'].apply(lambda words: len(words))

    text_data.fillna(0, inplace=True)

    text_click_data = pd.merge(user_interact_train,text_data,how='left',on=['photo_id'])

    print (text_click_data.info())

    # 原始被点击语料库
    corpus = []
    for cover_words in text_click_data['cover_words']:
        corpus.append(cover_words)

    # 词典
    DICTIONARY_PATH = '../sample/dictionary.txt' if USE_SAMPLE else '../data/dictionary.txt'
    if not os.path.exists(DICTIONARY_PATH):
        dictionary = corpora.Dictionary(corpus)
        dictionary.save(DICTIONARY_PATH)
    else:
        dictionary = corpora.Dictionary.load(DICTIONARY_PATH)

    corpus = [dictionary.doc2bow(text) for text in corpus]

    # tfidf模型
    TFIDF_MODEL_PATH = '../sample/tfidf_model.model' if USE_SAMPLE else '../data/tfidf_model.model'
    if not os.path.exists(TFIDF_MODEL_PATH):
        tfidf_model = models.TfidfModel(corpus)
        tfidf_model.save(TFIDF_MODEL_PATH)
    else:
        tfidf_model = models.TfidfModel.load(TFIDF_MODEL_PATH)
    corpus_tfidf = tfidf_model[corpus]

    key_words = []
    for item in corpus_tfidf:
        word_tfidf = sorted(item, key=lambda x: x[1], reverse=True)
        if len(word_tfidf) >= 2:
            key_words.extend([dictionary.get(i[0]) for i in word_tfidf[:2]])
        elif len(word_tfidf) == 1:
            key_words.extend([dictionary.get(word_tfidf[0][0])])

    count = Counter(key_words)
    print(len(count))
    # top_k关键词
    k = 2000
    top_key_words = count.most_common(k)
    top_key_words_word = [i[0] for i in top_key_words]

    def key_words_num(words):
        num = 0
        for word in words:
            if word in top_key_words_word:
                num += 1
        return num


    text_data['key_words_num'] = text_data['cover_words'].apply(key_words_num)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_df=0.7)
    corpus = text_data['cover_words'].apply(lambda words: ' '.join(words))
    tfidf = vectorizer.fit_transform(corpus)
    avg_tfidf = np.mean(tfidf, axis=1)
    text_data['avg_tfidf'] = avg_tfidf

    # WORD2VEC_MODEL_PATH = '../sample/word2vec.model' if USE_SAMPLE else '../data/word2vec.model'
    # word_embedding_corpus = []
    # if os.path.exists(WORD2VEC_MODEL_PATH):
    #     model = load_model(WORD2VEC_MODEL_PATH)
    #     for sentence in corpus:
    #         word_embedding = np.zeros(100,)
    #         for word in sentence:
    #             word_embedding += model.wv[word]
    #         word_embedding_corpus.append(word_embedding)
    # else:
    #     model = train_word2vec(corpus,WORD2VEC_MODEL_PATH)
    #     for sentence in corpus:
    #         word_embedding = np.zeros(100, )
    #         for word in sentence:
    #             word_embedding += model.wv[word]
    #         word_embedding_corpus.append(word_embedding)

    CLUSTER_MODEL_PATH = '../sample/doc_cluster.pkl' if USE_SAMPLE else '../data/doc_cluster.pkl'

    if os.path.exists(CLUSTER_MODEL_PATH):
        km = load_cluster_model(CLUSTER_MODEL_PATH)
        text_data['cluster_label'] = cluster_model_predict(km,tfidf)
    else:
        model = train_cluster_model(CLUSTER_MODEL_PATH,tfidf)
        text_data['cluster_label'] = model.predict(tfidf)

    text_data.drop(['cover_words'], axis=1, inplace=True)

    CLASSIFY_MODEL_PATH = '../sample/classify_model' if USE_SAMPLE else '../data/classify_model'
    if not os.path.exists(CLASSIFY_MODEL_PATH + '.bin'):
        raise LookupError("classify model not found,please run text_classify.py")
    else:
        classifier = fasttext.load_model(CLASSIFY_MODEL_PATH + '.bin', label_prefix='__label__')
    def word_classify(words):
        if words == '0':
            return -1
        else:
            words = words.replace(',',' ')
            predict =classifier.predict([words])
            if predict[0] == 'click':
                return 1
            else:
                return 0

    text_data['predict_label'] = text_data['cover_words_4_predict'].apply(word_classify)
    text_data.drop(['cover_words_4_predict'],axis=1,inplace =True)

    TEXT_FEATURE_FILE = 'text_feature'
    TEXT_FEATURE_FILE = TEXT_FEATURE_FILE + '_sample' + '.' + fmt if USE_SAMPLE else TEXT_FEATURE_FILE + '.' + fmt
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    if not os.path.exists(feature_store_path):
        os.mkdir(feature_store_path)
    store_data(text_data, os.path.join(feature_store_path, TEXT_FEATURE_FILE), fmt)
