#coding:utf8

import random
import pandas as pd
import functools
import argparse
import sys

sys.path.append("..")
from conf.modelconf import *
import matplotlib
matplotlib.use('Agg')


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', help='sample random seed, default 777', default=777)
parser.add_argument('-p', '--prop', help='sample propotion, default 0.33', default=0.33)
args = parser.parse_args()

random.seed(a=int(args.seed))

def user_sample(group, prop):
    n = group.shape[0]
    # this sample split add time split, make sure offline identical with online
    group = group.sort_values('time', ascending=False)
    m = int(prop * n)
    # print(m)
    m = 1 if m == 0 else m
    group = group[:m]
    # print(group.head())
    return group

def photo_sample(df, photo_ids):
    t = df[df['photo_id'].isin(photo_ids)]
    return t

print(get_root())

TRAIN_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_interaction.txt')
TEST_USER_INTERACT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_interaction.txt')

TRAIN_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_face.txt')
TEST_FACE, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_face.txt')

TRAIN_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('train_text.txt')
TEST_TEXT, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('test_text.txt')

TRAIN_VISUAL, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('visual_train')
TEST_VISUAL, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file('visual_test')



print(TRAIN_USER_INTERACT)
print(TEST_USER_INTERACT)
print(TRAIN_FACE)
print(TEST_FACE)
print(TRAIN_TEXT)
print(TEST_TEXT)
print(TRAIN_VISUAL)
print(TEST_VISUAL)

face_train = pd.read_csv(TRAIN_FACE,
                        sep='\t',
                        header=None,
                        names=['photo_id', 'faces'])

user_item_train = pd.read_csv(TRAIN_USER_INTERACT,
                             sep='\t',
                             header=None,
                             names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])

text_train = pd.read_csv(TRAIN_TEXT,
                       sep='\t',
                       header=None,
                       names=['photo_id', 'cover_words'])

print(float(args.prop))
users = list(user_item_train['user_id'].unique())
i = 0
user_sample = functools.partial(user_sample, prop=float(args.prop))
kfold_user_item_val = user_item_train.groupby(['user_id']).apply(user_sample)
kfold_user_item_val.reset_index(drop=True, inplace=True)
print(kfold_user_item_train.time.min())

kfold_user_item_train = user_item_train.append(kfold_user_item_val).drop_duplicates(keep=False).reset_index(drop=True)
print(kfold_user_item_train.time.max())

val_photo_ids = set(kfold_user_item_val['photo_id'].unique()) - set(kfold_user_item_train['photo_id'].unique())
inter_train_val_photo_ids = set(kfold_user_item_val['photo_id'].unique()) & set(kfold_user_item_train['photo_id'].unique())
# online: train_user_ids = test_user_ids; train_photo_ids & test_photo_ids = null; train_time.max < test_time.min;
# offline: this train will include train val intersection inter_train_val_photo_ids;
# satisfy: train_user_ids = val_user_ids; train_photo_ids & val_photo_ids = null; train_time.max < val_time.min;
# train_photo_ids = kfold_user_item_train['photo_id'].unique()
print("train shape: (%d, %d)" % (kfold_user_item_train.shape[0], kfold_user_item_train.shape[1]))
print("valid shape: (%d, %d)" % (kfold_user_item_val.shape[0], kfold_user_item_val.shape[1]))
kfold_val_users = kfold_user_item_val['user_id'].unique()
kfold_train_users = kfold_user_item_train['user_id'].unique()

print("val users: %s" % len(kfold_val_users))
print("train users: %s" % len(kfold_train_users))

print("valid photos before remove intersection: %s" % len(set(kfold_user_item_val['photo_id'].unique())))
print("valid photos after remove train/val intersection: %s" % len(val_photo_ids))
print('train click mean: %s' % user_item_train['click'].mean())
print('val click mean before remove intersection: %s' % kfold_user_item_val['click'].mean())
kfold_user_item_val_1 = kfold_user_item_val.loc[kfold_user_item_val.photo_id.isin(val_photo_ids)]
kfold_user_item_val_2 = kfold_user_item_val.loc[~kfold_user_item_val.photo_id.isin(val_photo_ids)]
# 这里取train_photo_ids 的话，就是训练集会加入更多的来自未做交集之前的验证集合的样例，但是无法保证时间完全先后顺序
min_time = kfold_user_item_val_1.time.min()
print(min_time)
kfold_user_item_val_2 = kfold_user_item_val_2.loc[kfold_user_item_val_2['time']<min_time]
print(kfold_user_item_val_2.shape)
print(kfold_user_item_train.time.max())
kfold_user_item_train = kfold_user_item_train.append(kfold_user_item_val_2)
print(kfold_user_item_train.time.max())
kfold_user_item_val = kfold_user_item_val_1
print('val click mean after remove intersection: %s' % kfold_user_item_val['click'].mean())
print("train shape after validation remove intersection: (%d, %d)" % (kfold_user_item_train.shape[0], kfold_user_item_train.shape[1]))
print("valid shape after validation remove intersection: (%d, %d)" % (kfold_user_item_val.shape[0], kfold_user_item_val.shape[1]))


# pos_kfold_user_item_val = kfold_user_item_val[kfold_user_item_val['click']==1]
# neg_kfold_user_item_val =  kfold_user_item_val[kfold_user_item_val['click']==0]
# # negative_sample_ratio = (1-ctr_o)*c_s/(ctr_o * u_s)
# ctr_o = user_item_train['click'].mean()
# c_s = pos_kfold_user_item_val.shape[0]
# u_s = neg_kfold_user_item_val.shape[0]
# print(ctr_o, c_s, u_s)
# negative_sample_ratio = (1 - ctr_o) * c_s / (ctr_o * u_s)
# print(negative_sample_ratio)
# def negative_sample(group):
#     # make sure validation time took after train
#     group = group.sort_values('time', ascending=False)
#     n = group.shape[0]
#     m = int(negative_sample_ratio * n)
#     m = n if m == 0 else m
#     group = group.iloc[random.sample(range(n), m)]
#     return group
# neg_kfold_user_item_val = neg_kfold_user_item_val.groupby(['user_id']).apply(negative_sample)
# kfold_user_item_val = pd.concat([pos_kfold_user_item_val, neg_kfold_user_item_val]).reset_index(drop=True)
# val_photo_ids = set(kfold_user_item_val['photo_id'].unique())
# print('val click mean after neg sample: %s' % kfold_user_item_val['click'].mean())
# print('train click mean after neg sample: %s' % kfold_user_item_train['click'].mean())
# print("train shape after validation neg sample: (%d, %d)" % (kfold_user_item_train.shape[0], kfold_user_item_train.shape[1]))
# print("valid shape after validation neg sample: (%d, %d)" % (kfold_user_item_val.shape[0], kfold_user_item_val.shape[1]))

user_item_val_path = os.path.join(offline_data_dir, 'test_interaction' + str(i) + '.txt')
kfold_user_item_val.to_csv(user_item_val_path, sep='\t', index=False, header=False)
print(kfold_user_item_val.info())
print('The %dth fold test_interaction extracted to %s' % (i, user_item_val_path))
user_item_train_path = os.path.join(offline_data_dir, 'train_interaction' + str(i) + '.txt')
kfold_user_item_train.to_csv(user_item_train_path, sep='\t', index=False, header=False)
print(kfold_user_item_train.info())
print('The %dth fold train_interaction extracted to %s' % (i, user_item_train_path))

# 这里取train_photo_ids 的话，就是训练集会加入更多的来自未做交集之前的验证集合的样例，但是无法保证时间完全先后顺序
train_photo_ids = set(kfold_user_item_train['photo_id'].unique())

kfold_face_val = photo_sample(face_train, val_photo_ids)
kfold_face_train = photo_sample(face_train, train_photo_ids)
face_test_path = os.path.join(offline_data_dir, 'test_face' + str(i) + '.txt')
kfold_face_val.to_csv(face_test_path, sep='\t', index=False, header=False)
print(kfold_face_val.info())
print('The %dth fold test_face extracted to %s' % (i, face_test_path))
face_train_path = os.path.join(offline_data_dir, 'train_face' + str(i) + '.txt')
kfold_face_train.to_csv(face_train_path, sep='\t', index=False, header=False)
print(kfold_face_train.info())
print('The %dth fold train_face extracted to %s' % (i, face_train_path))

kfold_text_val = photo_sample(text_train, val_photo_ids)
kfold_text_train = photo_sample(text_train, train_photo_ids)
text_test_path = os.path.join(offline_data_dir, 'test_text' + str(i) + '.txt')
kfold_text_val.to_csv(text_test_path, sep='\t', index=False, header=False)
print(kfold_text_val.info())
print('The %dth fold test_text extracted to %s' % (i, text_test_path))
text_train_path = os.path.join(offline_data_dir, 'train_text' + str(i) + '.txt')
kfold_text_train.to_csv(text_train_path, sep='\t', index=False, header=False)
print(kfold_text_train.info())
print('The %dth fold train_text extracted to %s' % (i, text_train_path))

train_text_photos = set(kfold_user_item_train['photo_id'].unique())
ui_photos = set(kfold_text_train['photo_id'].unique())
print(len(train_text_photos), len(ui_photos))
print(len(train_text_photos & ui_photos))