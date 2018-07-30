# coding:utf8
import os
import argparse
import sys
import gc

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append('..')

import pandas as pd
import numpy as np
from conf.modelconf import *
from common.utils import read_data


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
args = parser.parse_args()


def gen_sequence_doc(df, key, feat):
    def reduce_join(group):
        seq = ' '.join(group)
        return {'%s_doc' % key: seq}
    res = df[key].groupby(df[feat]).apply(reduce_join).unstack()
    return res


def gen_user_pos_neg_docs(df):
    datas = []
    df_pos = df[df['click']==1][['user_id', 'photo_id']].copy()
    df_neg = df[df['click']==0][['user_id', 'photo_id']].copy()
    df_pos['user_pos_docs'] = df_pos['user_id'].groupby(df_pos['photo_id']).transform(lambda x: ' '.join(map(str, x)))
    df_pos.drop_duplicates(['photo_id'], inplace=True)
    df_neg['user_neg_docs'] = df_neg['user_id'].groupby(df_neg['photo_id']).transform(lambda x: ' '.join(map(str, x)))
    df_neg.drop_duplicates(['photo_id'], inplace=True)

    user_docs = pd.merge(df, df_pos[['photo_id', 'user_pos_docs']], how='left', on=['photo_id'])
    user_docs = pd.merge(user_docs, df_neg[['photo_id', 'user_neg_docs']], how='left', on=['photo_id'])
    # for photo_id, group in df.groupby(['photo_id']):
    #     user_pos_docs = ' '.join(df[df['click']==1]['user_id'].astype(str))
    #     user_neg_docs = ' '.join(df[df['click']==0]['user_id'].astype(str))
    #     # print(user_pos_docs+' '+user_neg_docs)
    #     datas.append({'photo_id': photo_id, 'user_docs': user_pos_docs+' '+user_neg_docs})
    # return pd.DataFrame(datas)
    return user_docs

def gen_photo_pos_neg_docs(df):
    datas = []
    df_pos = df[df['click'] == 1][['user_id', 'photo_id']].copy()
    df_neg = df[df['click'] == 0][['user_id', 'photo_id']].copy()
    df_pos['photo_pos_docs'] = df_pos['photo_id'].groupby(df_pos['user_id']).transform(lambda x: ' '.join(map(str, x)))
    df_pos.drop_duplicates(['user_id'], inplace=True)
    df_neg['photo_neg_docs'] = df_neg['photo_id'].groupby(df_neg['user_id']).transform(lambda x: ' '.join(map(str, x)))
    df_neg.drop_duplicates(['user_id'], inplace=True)

    photo_docs = pd.merge(df, df_pos[['user_id', 'photo_pos_docs']], how='left', on=['user_id'])
    photo_docs = pd.merge(photo_docs, df_neg[['user_id', 'photo_neg_docs']], how='left', on=['user_id'])

    # for user_id, group in df.groupby(['user_id']):
    #     photo_pos_docs = ' '.join(df[df['click']==1]['photo_id'].astype(str))
    #     photo_neg_docs = ' '.join(df[df['click']==0]['photo_id'].astype(str))
    #     # print(photo_pos_docs+' '+photo_neg_docs)
    #     datas.append({'user_id': user_id, 'photo_docs': photo_pos_docs+' '+photo_neg_docs})
    # return pd.DataFrame(datas)
    return photo_docs


if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    kfold = int(args.offline_kfold)
    if args.online:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_interaction.txt')
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_interaction.txt')
        TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_text.txt',
                                                                                              args.online)
        TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_text.txt',
                                                                                             args.online)

        VISUAL_FEATURE_TRAIN_FILE = 'visual_feature_train' + '.' + fmt
        VISUAL_FEATURE_TEST_FILE = 'visual_feature_test' + '.' + fmt

    else:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)
        TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'train_text' + str(kfold) + '.txt', online=False)
        TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_text' + str(kfold) + '.txt', online=False)

        VISUAL_FEATURE_TRAIN_FILE = 'visual_feature_train' + str(kfold) + '.' + fmt
        VISUAL_FEATURE_TEST_FILE = 'visual_feature_test' + str(kfold) + '.' + fmt

    path = os.path.join(feature_store_dir, VISUAL_FEATURE_TRAIN_FILE)
    visual_train = read_data(path, fmt)
    path = os.path.join(feature_store_dir, VISUAL_FEATURE_TEST_FILE)
    visual_test = read_data(path, fmt)

    user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                                    sep='\t',
                                    usecols=[0, 1, 2],
                                    header=None,
                                    names=['user_id', 'photo_id','click'])

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


    text_train['cover_words'] = text_train['cover_words'].apply(words_to_list)
    text_test['cover_words'] = text_test['cover_words'].apply(words_to_list)
    text_train['cover_words'] = text_train['cover_words'].apply(lambda x: ' '.join(x))
    text_test['cover_words'] = text_test['cover_words'].apply(lambda  x: ' '.join(x))

    user_item_train = pd.merge(user_item_train, text_train, how='left', on=['photo_id'])
    user_item_test = pd.merge(user_item_test, text_test, how='left', on=['photo_id'])
    user_item_train = pd.merge(user_item_train, visual_train, how='left', on=['user_id', 'photo_id'])
    user_item_test = pd.merge(user_item_test, visual_test, how='left', on=['user_id', 'photo_id'])

    del text_train
    del text_test
    del visual_train
    del visual_test
    gc.collect()
    # user_item = pd.concat([user_item_train, user_item_test])
    # print(user_item['user_id'].min(), user_item['user_id'].max())
    # print(user_item['photo_id'].min(), user_item['photo_id'].max())

    # user_docs = gen_user_docs(user_item)
    # photo_docs = gen_photo_docs(user_item)


    def generate_doc(df, name, concat_name):
        res = df.astype(str).groupby(name)[concat_name].apply((lambda x: ' '.join(x))).reset_index()
        res.columns = [name, '%s_doc' % concat_name]
        return res

    y_label = 'click' # 千万不要用 user_item_train[user_item_train[['click']] == 1], 很多返回空，不要加多索引[]
    for feature in ['photo_id', 'photo_cluster_label']:
        temp_feature = user_item_train[user_item_train[y_label] == 1][['user_id', feature]]
        temp_feature[feature] = temp_feature[feature].astype(str)
        temp = temp_feature.groupby('user_id')[feature].agg(lambda x: ' '.join(x)).reset_index(name="pos_" + feature)

        logging.info("pos_" +  feature + " 去除重复")
        temp["pos_" + feature] = temp["pos_" + feature].apply(lambda x: x.split(' '))
        temp["pos_" + feature] = temp["pos_" + feature].apply(lambda x: list(set(x)))
        temp["pos_" + feature] = temp["pos_" + feature].apply(lambda x: ' '.join(x))

        print("pos_" + feature)
        print(temp.head())


        # merge
        user_item_train = pd.merge(user_item_train, temp, on='user_id', how='left')
        user_item_test = pd.merge(user_item_test, temp, on='user_id', how='left')
        user_item_train['pos_' + feature] = user_item_train['pos_' + feature].astype(str)
        user_item_test['pos_' + feature] = user_item_test['pos_' + feature].astype(str)


        temp_feature = user_item_train[user_item_train[y_label] == 0][['user_id', feature]]
        temp_feature[feature] = temp_feature[feature].astype(str)
        temp = temp_feature.groupby('user_id')[feature].agg(lambda x: ' '.join(x)).reset_index(name="neg_" + feature)

        logging.info("neg_" +  feature + " 去除重复")
        temp["neg_" + feature] = temp["neg_" + feature].apply(lambda x: x.split(' '))
        temp["neg_" + feature] = temp["neg_" + feature].apply(lambda x: list(set(x)))
        temp["neg_" + feature] = temp["neg_" + feature].apply(lambda x: ' '.join(x))
        print("pos_" + feature)
        print(temp.head())

        user_item_train = pd.merge(user_item_train, temp, on = 'user_id', how = 'left')
        user_item_test = pd.merge(user_item_test, temp, on='user_id', how='left')
        user_item_train['neg_' + feature] = user_item_train['neg_' + feature].astype(str)
        user_item_test['neg_' + feature] = user_item_test['neg_' + feature].astype(str)

        del temp_feature
        del temp
        gc.collect()


    concat_feature = 'photo_cluster_label'
    for feature in ['user_id']:
        temp_feature = user_item_train[user_item_train[y_label] == 1][[concat_feature, feature]]
        temp_feature[feature] = temp_feature[feature].astype(str)
        temp = temp_feature.groupby(concat_feature)[feature].agg(lambda x: ' '.join(x)).reset_index(name="pos_" + feature)

        logging.info("pos_" +  feature + " 去除重复")
        temp["pos_" + feature] = temp["pos_" + feature].apply(lambda x: x.split(' '))
        temp["pos_" + feature] = temp["pos_" + feature].apply(lambda x: list(set(x)))
        temp["pos_" + feature] = temp["pos_" + feature].apply(lambda x: ' '.join(x))
        # merge
        user_item_train = pd.merge(user_item_train, temp, on=concat_feature, how='left')
        user_item_test = pd.merge(user_item_test, temp, on=concat_feature, how='left')
        user_item_train['pos_' + feature] = user_item_train['pos_' + feature].astype(str)
        user_item_test['pos_' + feature] = user_item_test['pos_' + feature].astype(str)


        temp_feature = user_item_train[user_item_train[y_label] == 0][[concat_feature, feature]]
        temp_feature[feature] = temp_feature[feature].astype(str)
        temp = temp_feature.groupby(concat_feature)[feature].agg(lambda x: ' '.join(x)).reset_index(name="neg_" + feature)

        logging.info("neg_" +  feature + " 去除重复")
        temp["neg_" + feature] = temp["neg_" + feature].apply(lambda x: x.split(' '))
        temp["neg_" + feature] = temp["neg_" + feature].apply(lambda x: list(set(x)))
        temp["neg_" + feature] = temp["neg_" + feature].apply(lambda x: ' '.join(x))

        user_item_train = pd.merge(user_item_train, temp, on=concat_feature, how='left')
        user_item_test = pd.merge(user_item_test, temp, on=concat_feature, how='left')
        user_item_train['neg_' + feature] = user_item_train['neg_' + feature].astype(str)
        user_item_test['neg_' + feature] = user_item_test['neg_' + feature].astype(str)

        del temp_feature
        del temp
        gc.collect()

    user_item_train.reset_index(drop=True, inplace=True)
    user_item_test.reset_index(drop=True, inplace=True)


    print(user_item_train.info())
    print(user_item_train.head())

    y_label = 'click'
    # 去掉当前记录的属性
    logging.info("pos_user_id" + " 去除当前记录")
    user_item_train.loc[user_item_train[y_label]==1, 'pos_user_id'] += " " + user_item_train[user_item_train[y_label]==1]['user_id'].astype(str)
    user_item_train.loc[user_item_train[y_label]==1, 'pos_user_id'] = user_item_train[user_item_train[y_label]==1]['pos_user_id'].apply(lambda x: x.split(' '))
    user_item_train.loc[user_item_train[y_label]==1, 'pos_user_id'] = user_item_train[user_item_train[y_label]==1]['pos_user_id'].apply(lambda x: list(filter(lambda item: item != x[-1], x)))
    user_item_train.loc[user_item_train[y_label]==1, 'pos_user_id'] = user_item_train[user_item_train[y_label]==1]['pos_user_id'].apply(lambda x: ' '.join(x))

    logging.info("pos_photo_id" + " 去除当前记录")
    user_item_train.loc[user_item_train[y_label]==1, 'pos_photo_id'] += " " + user_item_train[user_item_train[y_label]==1]['photo_id'].astype(str)
    user_item_train.loc[user_item_train[y_label]==1, 'pos_photo_id'] = user_item_train[user_item_train[y_label]==1]['pos_photo_id'].apply(lambda x: x.split(' '))
    user_item_train.loc[user_item_train[y_label]==1, 'pos_photo_id'] = user_item_train[user_item_train[y_label]==1]['pos_photo_id'].apply(
        lambda x: list(filter(lambda item: item != x[-1], x)))
    user_item_train.loc[user_item_train[y_label]==1, 'pos_photo_id'] = user_item_train[user_item_train[y_label]==1]['pos_photo_id'].apply(lambda x: ' '.join(x))

    logging.info("pos_photo_cluster_label" + " 去除当前记录")
    user_item_train.loc[user_item_train[y_label]==1, 'pos_photo_cluster_label'] += " " + user_item_train[user_item_train[y_label]==1]['photo_cluster_label'].astype(str)
    user_item_train.loc[user_item_train[y_label]==1, 'pos_photo_cluster_label'] = user_item_train[user_item_train[y_label]==1]['pos_photo_cluster_label'].apply(lambda x: x.split(' '))
    user_item_train.loc[user_item_train[y_label]==1, 'pos_photo_cluster_label'] = user_item_train[user_item_train[y_label]==1]['pos_photo_cluster_label'].apply(
        lambda x: list(filter(lambda item: item != x[-1], x)))
    user_item_train.loc[user_item_train[y_label]==1, 'pos_photo_cluster_label'] = user_item_train[user_item_train[y_label]==1]['pos_photo_cluster_label'].apply(lambda x: ' '.join(x))





    logging.info("neg_user_id" + " 去除当前记录")
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_user_id'] += " " + user_item_train[user_item_train[y_label]==0]['user_id'].astype(str)
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_user_id'] = user_item_train[user_item_train[y_label] == 0][
        'neg_user_id'].apply(lambda x: x.split(' '))
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_user_id'] = user_item_train[user_item_train[y_label] == 0][
        'neg_user_id'].apply(lambda x: list(filter(lambda item: item != x[-1], x)))
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_user_id'] = user_item_train[user_item_train[y_label] == 0][
        'neg_user_id'].apply(lambda x: ' '.join(x))

    logging.info("neg_photo_id" + " 去除当前记录")
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_photo_id'] += " " + user_item_train[user_item_train[y_label]==0]['photo_id'].astype(str)
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_photo_id'] = user_item_train[user_item_train[y_label] == 0][
        'neg_photo_id'].apply(lambda x: x.split(' '))
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_photo_id'] = user_item_train[user_item_train[y_label] == 0][
        'neg_photo_id'].apply(
        lambda x: list(filter(lambda item: item != x[-1], x)))
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_photo_id'] = user_item_train[user_item_train[y_label] == 0][
        'neg_photo_id'].apply(lambda x: ' '.join(x))


    logging.info("neg_photo_cluster_label" + " 去除当前记录")
    user_item_train.loc[
        user_item_train[y_label] == 0, 'neg_photo_cluster_label'] += " " + user_item_train[user_item_train[y_label]==0]['photo_cluster_label'].astype(str)
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_photo_cluster_label'] = \
    user_item_train[user_item_train[y_label] == 0]['neg_photo_cluster_label'].apply(lambda x: x.split(' '))
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_photo_cluster_label'] = \
    user_item_train[user_item_train[y_label] == 0]['neg_photo_cluster_label'].apply(
        lambda x: list(filter(lambda item: item != x[-1], x)))
    user_item_train.loc[user_item_train[y_label] == 0, 'neg_photo_cluster_label'] = \
    user_item_train[user_item_train[y_label] == 0]['neg_photo_cluster_label'].apply(lambda x: ' '.join(x))

    # user_docs = gen_user_pos_neg_docs(user_item_train)
    # photo_docs = gen_photo_pos_neg_docs(user_item_train)
    #
    # user_item_train = pd.merge(user_item_train, user_docs, how='left', on=['user_id','photo_id'])
    # user_item_train = pd.merge(user_item_train, photo_docs, how='left', on=['user_id', 'photo_id'])
    #
    # user_item_test = pd.merge(user_item_test, user_docs, how='left', on=['user_id', 'photo_id'])
    # user_item_test = pd.merge(user_item_test, photo_docs, how='left', on=['user_id', 'photo_id'])



    user_item_train.sort_values(['user_id', 'photo_id'], inplace=True)
    user_item_test.sort_values(['user_id', 'photo_id'], inplace=True)
    user_item_train.reset_index(drop=True, inplace=True)
    user_item_test.reset_index(drop=True, inplace=True)
    print(user_item_train.head())
    print(user_item_test.head())

    cv = CountVectorizer(token_pattern='\w+', max_features=20000)  # max_features = 20000
    train_matrixs = []
    test_matrixs = []
    print("开始cv.....")
    for feature in ['pos_user_id', 'neg_user_id', 'pos_photo_id', 'neg_photo_id', 'pos_photo_cluster_label', 'neg_photo_cluster_label', 'cover_words']:
        print("生成 " + feature + " CountVector")
        cv.fit(user_item_train[feature].astype('str'))
        print("开始转换 " + feature + " CountVector")
        train_temp = cv.transform(user_item_train[feature].astype('str'))
        test_temp = cv.transform(user_item_test[feature].astype('str'))
        train_matrixs.append(train_temp)
        test_matrixs.append(test_temp)

        print(feature + " is over")
    all_train_data_x = sparse.hstack(train_matrixs)
    test_data_x = sparse.hstack(test_matrixs)

    print('cv prepared')
    print(all_train_data_x.head())
    print(test_data_x.head())

    if args.online:
        sparse.save_npz(os.path.join(feature_store_dir, 'embedding_vector_feature_train.npz'), all_train_data_x)
        sparse.save_npz(os.path.join(feature_store_dir, 'embedding_vector_feature_test.npz'), test_data_x)
    else:
        sparse.save_npz(os.path.join(feature_store_dir, 'embedding_vector_feature_train' + str(kfold) + '.npz'), all_train_data_x)
        sparse.save_npz(os.path.join(feature_store_dir, 'embedding_vector_feature_test' + str(kfold) + '.npz'), test_data_x)


