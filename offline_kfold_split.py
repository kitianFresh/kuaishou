import random
import pandas as pd
import functools
import argparse
import sys

sys.path.append("..")
from conf.modelconf import *


parser = argparse.ArgumentParser()
parser.add_argument('-k', '--kfold', help='split data by kfold for offline validation, this split make sure offline data has the same distribution with online, it will be used for kfold feature statistics to avoid data leakage', default=3)
args = parser.parse_args()


def user_sample(group, prop):
    n = group.shape[0]
    m = int(prop * n)
    m = n if m == 0 else m
    group = group.iloc[random.sample(range(n), m)]
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

kfold = int(args.kfold)
prop = 1. / kfold
print(prop)
users = list(user_item_train['user_id'].unique())
m = int(len(users) * prop)
seeds = [2018*i for i in range(kfold)]
kfolds = []
for i in range(kfold):
    random.seed(a=seeds[i])
    user_sample = functools.partial(user_sample, prop=prop)
    kfold_user_item_val = user_item_train.groupby(['user_id']).apply(user_sample)
    kfold_user_item_val.reset_index(drop=True, inplace=True)
    kfold_users = kfold_user_item_val['user_id'].unique()
    kfold_user_item_train = user_item_train.append(kfold_user_item_val).drop_duplicates(keep=False).reset_index(drop=True)
    val_photo_ids = list(set(kfold_user_item_val['photo_id'].unique()) - set(kfold_user_item_train['photo_id'].unique()))
    print(len(kfold_users))
    print(len(val_photo_ids))
    kfolds.append((kfold_users, set(val_photo_ids)))

    kfold_user_item_val = kfold_user_item_val.loc[kfold_user_item_val.photo_id.isin(val_photo_ids)]
    kfold_user_item_train = user_item_train.loc[~user_item_train.photo_id.isin(val_photo_ids)]

    kfold_user_item_val.to_csv(os.path.join(offline_data_dir, 'test_interaction' + str(i) + '.txt'), sep='\t', index=False, header=False)
    print(kfold_user_item_val.info())
    print('The %dth fold test_interaction extracted' % i)
    kfold_user_item_train.to_csv(os.path.join(offline_data_dir, 'train_interaction' + str(i) + '.txt'), sep='\t', index=False, header=False)
    print(kfold_user_item_train.info())
    print('The %dth fold train_interaction extracted' % i)

    train_photo_ids = set(kfold_user_item_train['photo_id'].unique()) - set(val_photo_ids)

    kfold_face_val = photo_sample(face_train, val_photo_ids)
    kfold_face_train = photo_sample(face_train, train_photo_ids)
    kfold_face_val.to_csv(os.path.join(offline_data_dir, 'test_face' + str(i) + '.txt'), sep='\t', index=False, header=False)
    print(kfold_face_val.info())
    print('The %dth fold test_face extracted' % i)
    kfold_face_train.to_csv(os.path.join(offline_data_dir, 'train_face' + str(i) + '.txt'), sep='\t', index=False, header=False)
    print(kfold_face_train.info())
    print('The %dth fold train_face extracted' % i)

    kfold_text_val = photo_sample(text_train, val_photo_ids)
    kfold_text_train = photo_sample(text_train, train_photo_ids)
    kfold_text_val.to_csv(os.path.join(offline_data_dir, 'test_text' + str(i) + '.txt'), sep='\t', index=False, header=False)
    print(kfold_text_val.info())
    print('The %dth fold test_text extracted' % i)
    kfold_text_train.to_csv(os.path.join(offline_data_dir, 'train_text' + str(i) + '.txt'), sep='\t', index=False, header=False)
    print(kfold_text_train.info())
    print('The %dth fold train_text extracted' % i)

import matplotlib.pyplot as plt
from matplotlib_venn import venn3
photo_sets = []
for i in range(kfold):
    kfold_users, val_photo_ids = kfolds[i]
    photo_sets.append(val_photo_ids)
    print(len(kfold_users), len(val_photo_ids))

venn3(photo_sets, ('Fold%s' % str(i) for i in range(kfold)))
# plt.show()
plt.savefig('%sFoldSet' % kfold)
