# coding:utf8
import os
import argparse
import sys
import gc

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append('..')

# import modin.pandas as pd
import pandas as pd
import numpy as np

from tqdm import tqdm
from collections import defaultdict

from conf.modelconf import *
from common.utils import read_data


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-o', '--online', help='online feature extract', action="store_true")
parser.add_argument('-k', '--offline-kfold', help='offline kth fold feature extract, extract kth fold', default=0)
parser.add_argument('-c', '--column', help='type name of count vector feature you want to extract')
parser.add_argument('-m', '--max-feature', help='max feature to use count vector', default=500000)
args = parser.parse_args()


def digitize():
    uid_aid_train = pd.read_csv('dataset/train_uid_aid.csv')
    uid_aid_test1 = pd.read_csv('dataset/test1_uid_aid.csv')
    uid_aid_test2 = pd.read_csv('dataset/test2_uid_aid.csv')
    uid_aid_df = pd.concat([uid_aid_train, uid_aid_test1, uid_aid_test2], axis=0)
    for col in range(3):
        bins = []
        for percent in [0, 20, 35, 50, 65, 85, 100]:
            bins.append(np.percentile(uid_aid_df.iloc[:, col], percent))
        uid_aid_train.iloc[:, col] = np.digitize(uid_aid_train.iloc[:, col], bins, right=True)
        uid_aid_test1.iloc[:, col] = np.digitize(uid_aid_test1.iloc[:, col], bins, right=True)
        uid_aid_test2.iloc[:, col] = np.digitize(uid_aid_test2.iloc[:, col], bins, right=True)

    count_bins = [1, 2, 4, 6, 8, 10, 16, 27, 50]
    uid_aid_train.iloc[:, 3] = np.digitize(uid_aid_train.iloc[:, 3], count_bins, right=True)
    uid_aid_test1.iloc[:, 3] = np.digitize(uid_aid_test1.iloc[:, 3], count_bins, right=True)
    uid_aid_test2.iloc[:, 3] = np.digitize(uid_aid_test2.iloc[:, 3], count_bins, right=True)

    uid_convert_train = pd.read_csv("dataset/train_neg_pos_aid.csv", usecols=['uid_convert'])
    uid_convert_test2 = pd.read_csv("dataset/test2_neg_pos_aid.csv", usecols=['uid_convert'])

    convert_bins = [-1, 0, 0.1, 0.3, 0.5, 0.7, 1]
    uid_convert_train.iloc[:, 0] = np.digitize(uid_convert_train.iloc[:, 0], convert_bins, right=True)
    uid_convert_test2.iloc[:, 0] = np.digitize(uid_convert_test2.iloc[:, 0], convert_bins, right=True)

    uid_aid_train = pd.concat([uid_aid_train, uid_convert_train], axis=1)
    uid_aid_test2 = pd.concat([uid_aid_test2, uid_convert_test2], axis=1)

    uid_aid_train.to_csv('dataset/train_uid_aid_bin.csv', index=False)
    uid_aid_test2.to_csv('dataset/test2_uid_aid_bin.csv', index=False)


def gen_pos_neg_pid_fea(train_data, test2_data):

    train_user = train_data.user_id.unique()

    # user-aid dict
    uid_dict = defaultdict(list)
    for row in tqdm(train_data.iterrows(), total=len(train_data)):
        uid_dict[row['user_id']].append([row['photo_id'], row['click']])

    # user convert
    uid_convert = {}
    for uid in tqdm(train_user):
        pos_pid, neg_pid = [], []
        for data in uid_dict[uid]:
            if data[1] == 1:
                pos_pid.append(data[0])
            else:
                neg_pid.append(data[0])
        uid_convert[uid] = [pos_pid, neg_pid]

    test2_neg_pos_pid = {}
    for row in tqdm(test2_data.iterrows(), total=len(test2_data)):
        pid = row['photo_id']
        uid = row['user_id']
        if uid_convert.get(uid, []) == []:
            test2_neg_pos_pid[row[0]] = ['', '', -1]
        else:
            pos_pid, neg_pid = uid_convert[uid][0].copy(), uid_convert[uid][1].copy()
            convert = len(pos_pid) / (len(pos_pid) + len(neg_pid)) if (len(pos_pid) + len(neg_pid)) > 0 else -1
            test2_neg_pos_pid[row['user_id']] = [' '.join(map(str, pos_pid)), ' '.join(map(str, neg_pid)), convert]
    df_test2 = pd.DataFrame.from_dict(data=test2_neg_pos_pid, orient='index')
    df_test2.columns = ['pos_photo_id', 'neg_photo_id', 'uid_convert']

    train_neg_pos_pid = {}
    for row in tqdm(train_data.itertuples(), total=len(train_data)):
        pid = row['photo_id']
        uid = row['user_id']
        pos_pid, neg_pid = uid_convert[uid][0].copy(), uid_convert[uid][1].copy()
        if pid in pos_pid:
            pos_pid.remove(pid)
        if pid in neg_pid:
            neg_pid.remove(pid)
        convert = len(pos_pid) / (len(pos_pid) + len(neg_pid)) if (len(pos_pid) + len(neg_pid)) > 0 else -1
        train_neg_pos_pid[row['user_id']] = [' '.join(map(str, pos_pid)), ' '.join(map(str, neg_pid)), convert]

    df_train = pd.DataFrame.from_dict(data=train_neg_pos_pid, orient='index')
    df_train.columns = ['pos_photo_id', 'neg_photo_id', 'uid_convert']

    return df_train, df_test2


# def gen_pos_neg_id_fea(train_data, test2_data, who, id_feat):
#
#     whos = train_data[who].unique()
#
#     # user-aid dict
#     whos_dict = defaultdict(list)
#     for _, row in train_data.iterrows():
#         whos_dict[row[who]].append([row[id_feat], row['click']])
#
#     # user convert
#     whos_convert = {}
#     for id in whos:
#         pos_id, neg_id = [], []
#         for data in whos_dict[id]:
#             if data[1] == 1:
#                 pos_id.append(data[0])
#             else:
#                 neg_id.append(data[0])
#         whos_convert[id] = [pos_id, neg_id]
#
#     test2_neg_pos_id = {}
#     for _, row in test2_data.iterrows():
#         id = row[id_feat]
#         whoid = row[who]
#         if whos_convert.get(whoid, []) == []:
#             test2_neg_pos_id[row[0]] = ['', '', -1]
#         else:
#             # AttributeError: 'list' object has no attribute 'copy' python2 use list([]), python new method for copy list, copy
#             pos_id, neg_id = whos_convert[whoid][0].copy(), whos_convert[whoid][1].copy()
#             convert = len(pos_id) / (len(pos_id) + len(neg_id)) if (len(pos_id) + len(neg_id)) > 0 else -1
#             test2_neg_pos_id[row[who]] = [' '.join(map(str, pos_id)), ' '.join(map(str, neg_id)), convert]
#     df_test2 = pd.DataFrame.from_dict(data=test2_neg_pos_id, orient='index')
#     df_test2.columns = ['pos_' + id_feat, 'neg_' + id_feat, who + '_ctr']
#
#     train_neg_pos_id = {}
#     for _, row in train_data.itertuples.iterrows():
#         id = row[id_feat]
#         whoid = row[who]
#         pos_id, neg_id = whos_convert[whoid][0].copy(), whos_convert[whoid][1].copy()
#         if id in pos_id:
#             pos_id.remove(id)
#         if id in neg_id:
#             neg_id.remove(id)
#         convert = len(pos_id) / (len(pos_id) + len(neg_id)) if (len(pos_id) + len(neg_id)) > 0 else -1
#         train_neg_pos_id[row[who]] = [' '.join(map(str, pos_id)), ' '.join(map(str, neg_id)), convert]
#
#     df_train = pd.DataFrame.from_dict(data=train_neg_pos_id, orient='index')
#     df_train.columns = ['pos_' + id_feat, 'neg_' + id_feat, who + '_ctr']
#
#     return df_train, df_test2


def gen_pos_neg_id_fea(train_data, test2_data, who, id_feat):

    whos = train_data[who].unique()

    # user-aid dict
    whos_dict = defaultdict(list)
    for row in tqdm(train_data.itertuples(), total=len(train_data)):
        whos_dict[getattr(row, who)].append([getattr(row, id_feat), getattr(row, 'click')])

    # user convert
    whos_convert = {}
    for id in whos:
        pos_id, neg_id = [], []
        for data in whos_dict[id]:
            if data[1] == 1:
                pos_id.append(data[0])
            else:
                neg_id.append(data[0])
        whos_convert[id] = [pos_id, neg_id]

    test2_neg_pos_id = {}
    for row in tqdm(test2_data.itertuples(), total=len(test2_data)):
        id = getattr(row, id_feat)
        whoid = getattr(row, who)
        if whos_convert.get(whoid, []) == []:
            test2_neg_pos_id[row[0]] = ['', '', -1]
        else:
            # AttributeError: 'list' object has no attribute 'copy' python2 use list([]), python new method for copy list, copy
            pos_id, neg_id = list(whos_convert[whoid][0]), list(whos_convert[whoid][1])
            pos_len, neg_len = len(pos_id), len(neg_id)
            total = (pos_len + neg_len)*1.
            convert = pos_len / total if total > 0 else -1
            test2_neg_pos_id[whoid] = [' '.join(map(str, pos_id)), ' '.join(map(str, neg_id)), convert]
#     df_test2 = pd.DataFrame.from_dict(data=test2_neg_pos_id, orient='index')
    df_test2 = pd.DataFrame.from_dict(data=test2_neg_pos_id, orient='index').reset_index()
    df_test2.columns = [who, 'pos_' + id_feat, 'neg_' + id_feat, who + '_ctr']

    train_neg_pos_id = {}
    for row in tqdm(train_data.itertuples(), total=len(train_data)):
        id = getattr(row, id_feat)
        whoid = getattr(row, who)
        pos_id, neg_id = list(whos_convert[whoid][0]), list(whos_convert[whoid][1])
        if id in pos_id:
            pos_id.remove(id)
        if id in neg_id:
            neg_id.remove(id)
        pos_len, neg_len = len(pos_id), len(neg_id)
        total = (pos_len + neg_len)*1.
        convert = pos_len / total if total > 0 else -1
        train_neg_pos_id[whoid] = [' '.join(map(str, pos_id)), ' '.join(map(str, neg_id)), convert]

    df_train = pd.DataFrame.from_dict(data=train_neg_pos_id, orient='index').reset_index()
    df_train.columns = [who, 'pos_' + id_feat, 'neg_' + id_feat, who + '_ctr']

    return df_train, df_test2


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



    user_item_train = pd.merge(user_item_train, visual_train, how='left', on=['user_id', 'photo_id'])
    user_item_test = pd.merge(user_item_test, visual_test, how='left', on=['user_id', 'photo_id'])

    del visual_train
    del visual_test
    gc.collect()

    id_features = [args.column]
    y_label = 'click' # 千万不要用 user_item_train[user_item_train[['click']] == 1], 很多返回空，不要加多索引[]
    who = 'user_id'
    for id_feature in id_features:
        train, test = gen_pos_neg_id_fea(user_item_train, user_item_test, who, id_feature)
        print(train.head())
        print(test.head())
        user_item_train = pd.merge(user_item_train, train, how='left', on=[who])
        user_item_test = pd.merge(user_item_test, test, how='left', on=[who])


    # who = 'photo_cluster_label'
    # for id_feature in ['user_id']:
    #     train, test = gen_pos_neg_id_fea(user_item_train, user_item_test, who, id_feature)
    #     print(train.head())
    #     print(test.head())
    #     user_item_train = pd.merge(user_item_train, train, how='left', on=[who])
    #     user_item_test = pd.merge(user_item_test, test, how='left', on=[who])

    user_item_train.reset_index(drop=True, inplace=True)
    user_item_test.reset_index(drop=True, inplace=True)


    print(user_item_train.info())
    print(user_item_train.head())


    user_item_train.sort_values(['user_id', 'photo_id'], inplace=True)
    user_item_test.sort_values(['user_id', 'photo_id'], inplace=True)
    user_item_train.reset_index(drop=True, inplace=True)
    user_item_test.reset_index(drop=True, inplace=True)
    print(user_item_train.head())
    print(user_item_test.head())

    cv = CountVectorizer(token_pattern='\w+', max_features=int(args.max_feature))  # max_features = 20000
    train_matrixs = []
    test_matrixs = []
    print("开始cv.....")
    # pos_user_id, neg_user_id
    pos_id_features = ['pos_' + feat for feat in id_features]
    neg_id_features = ['neg_' + feat for feat in id_features]
    for feature in pos_id_features+neg_id_features:
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
    print(all_train_data_x.shape)
    print(test_data_x.shape)

    for feat in id_features:
        if args.online:
            sparse.save_npz(os.path.join(feature_store_dir, feat + '_vector_feature_train.npz'), all_train_data_x)
            sparse.save_npz(os.path.join(feature_store_dir, feat + '_vector_feature_test.npz'), test_data_x)
        else:
            sparse.save_npz(os.path.join(feature_store_dir, feat + '_vector_feature_train' + str(kfold) + '.npz'), all_train_data_x)
            sparse.save_npz(os.path.join(feature_store_dir, feat + '_vector_feature_test' + str(kfold) + '.npz'), test_data_x)


