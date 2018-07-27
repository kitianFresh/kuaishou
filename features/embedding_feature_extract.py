# coding:utf8
import os
import argparse
import sys

sys.path.append('..')

import pandas as pd
import numpy as np
from gensim import corpora, models
from collections import Counter
from conf.modelconf import *


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
    df_pos['pos_docs'] = df_pos['user_id'].groupby(df_pos['photo_id']).transform(lambda x: ' '.join(map(str, x)))
    df_pos.drop_duplicates(['photo_id'], inplace=True)
    df_neg['neg_docs'] = df_neg['user_id'].groupby(df_neg['photo_id']).transform(lambda x: ' '.join(map(str, x)))
    df_neg.drop_duplicates(['photo_id'], inplace=True)

    user_docs = pd.merge(df_pos[['photo_id', 'pos_docs']], df_neg[['photo_id', 'neg_docs']], how='inner', on=['photo_id'])

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
    df_pos['pos_docs'] = df_pos['photo_id'].groupby(df_pos['user_id']).transform(lambda x: ' '.join(map(str, x)))
    df_pos.drop_duplicates(['user_id'], inplace=True)
    df_neg['neg_docs'] = df_neg['photo_id'].groupby(df_neg['user_id']).transform(lambda x: ' '.join(map(str, x)))
    df_neg.drop_duplicates(['user_id'], inplace=True)

    photo_docs = pd.merge(df_pos[['user_id', 'pos_docs']], df_neg[['user_id', 'neg_docs']], how='inner', on=['user_id'])

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

    else:
        TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'train_interaction' + str(kfold) + '.txt', online=False)
        TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
            'test_interaction' + str(kfold) + '.txt', online=False)

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


    # user_item = pd.concat([user_item_train, user_item_test])
    # print(user_item['user_id'].min(), user_item['user_id'].max())
    # print(user_item['photo_id'].min(), user_item['photo_id'].max())

    # user_docs = gen_user_docs(user_item)
    # photo_docs = gen_photo_docs(user_item)


    def generate_doc(df, name, concat_name):
        res = df.astype(str).groupby(name)[concat_name].apply((lambda x: ' '.join(x))).reset_index()
        res.columns = [name, '%s_doc' % concat_name]
        return res


    user_docs = gen_user_pos_neg_docs(user_item_train)
    photo_docs = gen_photo_pos_neg_docs(user_item_train)

    print(user_docs.head())
    print(photo_docs.head())
