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
from text_cluster import train_cluster_model,cluster_model_predict
from word_embedding import load_model,train_word2vec
from conf.modelconf import get_data_file, data_dir, feature_dtype_map
from common.utils import count_feat_ctr

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
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
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_interaction.txt')


    else:
        TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_text' + str(kfold) + '.txt', online=False)
        TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_text' + str(kfold) + '.txt', online=False)

        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'test_interaction' + str(kfold) + '.txt', online=False)

    # trained once, use anywhere, shared for all online or offline folds
    DICTIONARY_PATH = os.path.join(data_dir, 'dictionary.txt')
    TFIDF_MODEL_PATH = os.path.join(data_dir, 'tfidf_model.model')
    WORD2VEC_MODEL_PATH = os.path.join(data_dir, 'word2vec.model')
    CLUSTER_MODEL_PATH = os.path.join(data_dir, 'doc_cluster')
    CLASSIFY_MODEL_PATH = os.path.join(data_dir, 'classify_model')

    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                                  sep='\t',
                                  usecols=[0, 1, 2],
                                  header=None,
                                  names=['user_id', 'photo_id', 'click'])

    print(user_item_train.info())

    user_item_test = pd.read_csv(TEST_USER_INTERACT,
                                 sep='\t',
                                 usecols=[0, 1],
                                 header=None,
                                 names=['user_id', 'photo_id'])

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

    def words_to_list(words):
        if words == '0':
            return []
        else:
            return words.split(',')

    text_train['cover_words_4_predict'] = text_train['cover_words']
    text_train['cover_words'] = text_train['cover_words'].apply(words_to_list)
    text_train['cover_length'] = text_train['cover_words'].apply(lambda words: len(words))

    text_test['cover_words_4_predict'] = text_test['cover_words']
    text_test['cover_words'] = text_test['cover_words'].apply(words_to_list)
    text_test['cover_length'] = text_test['cover_words'].apply(lambda words: len(words))

    p1 = set(text_train['photo_id'].unique())
    p2 = set(user_item_train['photo_id'].unique())
    print(len(p1), len(p2))
    print(len(p1 & p2))

    print(text_train.info())
    print(text_test.info())
    print(np.sum(text_train.isnull()))
    print(np.sum(text_test.isnull()))

    text_data = pd.concat([text_train, text_test])




    # 原始语料库
    corpus = []
    for cover_words in text_data['cover_words']:
        corpus.append(cover_words)

    # 词典
    if not os.path.exists(DICTIONARY_PATH):
        dictionary = corpora.Dictionary(corpus)
        dictionary.save(DICTIONARY_PATH)
    else:
        dictionary = corpora.Dictionary.load(DICTIONARY_PATH)

    corpus_index = [dictionary.doc2bow(text) for text in corpus]

    # tfidf模型
    if not os.path.exists(TFIDF_MODEL_PATH):
        tfidf_model = models.TfidfModel(corpus_index)
        tfidf_model.save(TFIDF_MODEL_PATH)
    else:
        tfidf_model = models.TfidfModel.load(TFIDF_MODEL_PATH)
    corpus_tfidf = tfidf_model[corpus_index]

    key_words = []
    avg_tfidf = []
    for item in corpus_tfidf:
        word_tfidf = sorted(item, key=lambda x: x[1], reverse=True)
        word_tfidf_val = [i[1] for i in word_tfidf]
        if len(word_tfidf_val) == 0 :
            avg_tfidf.append(0)
        else:
            avg_tfidf.append(np.mean(word_tfidf_val))
        if len(word_tfidf) >= 2:
            key_words.extend([dictionary.get(i[0]) for i in word_tfidf[:2]])
        elif len(word_tfidf) == 1:
            key_words.extend([dictionary.get(word_tfidf[0][0])])

    count = Counter(key_words)
    print(len(count))
    # top_k关键词
    k = 2000
    top_key_words = count.most_common(k)
    top_key_words = [i[0] for i in top_key_words]

    def key_words_num(words):
        num = 0
        for word in words:
            if word in top_key_words:
                num += 1
        return num


    text_data['key_words_num'] = text_data['cover_words'].apply(key_words_num)
    text_data['avg_tfidf'] = avg_tfidf


    word_embedding_corpus = []
    if os.path.exists(WORD2VEC_MODEL_PATH):
        model = load_model(WORD2VEC_MODEL_PATH)
        for sentence in corpus:
            word_embedding = np.zeros(100,)
            for word in sentence:
                try:
                    word_embedding += model.wv[word]
                except:
                    continue
            word_embedding_corpus.append(word_embedding)
    else:
        model = train_word2vec(corpus,WORD2VEC_MODEL_PATH)
        for sentence in corpus:
            word_embedding = np.zeros(100, )
            for word in sentence:
                try:
                    word_embedding += model.wv[word]
                except:
                    continue
            word_embedding_corpus.append(word_embedding)
    word_embedding_corpus = np.array(word_embedding_corpus)

    cluster_nums = 20
    if os.path.exists(CLUSTER_MODEL_PATH + '_kmeans' + str(cluster_nums) + '.pkl'):
        text_data['text_cluster_label'] = cluster_model_predict(CLUSTER_MODEL_PATH,word_embedding_corpus, cluster_nums)
    else:
        text_data['text_cluster_label'] = train_cluster_model(CLUSTER_MODEL_PATH,word_embedding_corpus,cluster_nums)


    text_data.drop(['cover_words'], axis=1, inplace=True)

    if not os.path.exists(CLASSIFY_MODEL_PATH + '.bin'):
        print("classify model not found, run text_classify.py")
        if args.online:
            os.system("python text_classify.py -k %s -o" % (args.offline_kfold))
        else:
            os.system("python text_classify.py -k %s" % (args.offline_kfold))
    print("*" * 60)
    print(CLASSIFY_MODEL_PATH)
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

    text_data['text_class_label'] = text_data['cover_words_4_predict'].apply(word_classify)
    text_data.drop(['cover_words_4_predict'],axis=1,inplace =True)

    text_data['have_text_cate'] = text_data['cover_length'].apply(lambda x: x > 0).astype(feature_dtype_map['have_text_cate'])
    text_data['text_class_label'] = text_data['text_class_label'].astype(feature_dtype_map['text_class_label'])
    text_data['text_cluster_label'] = text_data['text_cluster_label'].astype(feature_dtype_map['text_cluster_label'])


    if args.online:
        TEXT_FEATURE_FILE = 'text_feature' + '.' + fmt
    else:
        TEXT_FEATURE_FILE = 'text_feature' + str(kfold) + '.' + fmt

    print(text_data.info())
    if not os.path.exists(feature_store_dir):
        os.mkdir(feature_store_dir)
    store_data(text_data, os.path.join(feature_store_dir, TEXT_FEATURE_FILE), fmt)