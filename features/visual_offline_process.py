
import os
import sys
sys.path.append('../')

import pandas as pd

from conf.modelconf import *
online_feature_store_dir = os.path.join(online_data_dir, 'features')
visual_train_file = os.path.join(online_feature_store_dir, 'visual_feature_train.csv')
visual_test_file = os.path.join(online_feature_store_dir, 'visual_feature_test.csv')

inter_train_file = os.path.join(online_data_dir, 'train_interaction.txt')
inter_test_file = os.path.join(online_data_dir, 'test_interaction.txt')

feature_store_dir = os.path.join(offline_data_dir, 'features')

inter_train_file0 = os.path.join(offline_data_dir, 'train_interaction0.txt')
inter_test_file0 = os.path.join(offline_data_dir, 'test_interaction0.txt')
print(inter_train_file0)


user_item_train = pd.read_csv(inter_train_file,
                             sep='\t',
                             header=None,
                             names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])

user_item_train0 = pd.read_csv(inter_train_file0,
                             sep='\t',
                             header=None,
                             names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])


user_item_test = pd.read_csv(inter_test_file,
                             sep='\t',
                             header=None,
                             names=['user_id', 'photo_id', 'time', 'duration_time'])

user_item_test0 = pd.read_csv(inter_test_file0,
                             sep='\t',
                             header=None,
                            names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])



user_item_train.sort_values(['user_id', 'photo_id'], inplace=True)
user_item_test.sort_values(['user_id', 'photo_id'], inplace=True)
user_item_train0.sort_values(['user_id', 'photo_id'], inplace=True)
user_item_test0.sort_values(['user_id', 'photo_id'], inplace=True)


visual_train = pd.read_csv(visual_train_file, sep='\t')
visual_test = pd.read_csv(visual_test_file, sep='\t')
print(user_item_train0.head())
print(user_item_test0.head())
visual_sample_train0 = pd.merge(user_item_train0, visual_train, how='left', on=['user_id', 'photo_id'])
visual_sample_test0 = pd.merge(user_item_test0, visual_train, how='left', on=['user_id', 'photo_id'])
visual_sample_train0[['user_id', 'photo_id', 'photo_cluster_label']].to_csv(os.path.join(feature_store_dir,'visual_feature_train0.csv'), sep='\t', index=False)
visual_sample_test0[['user_id', 'photo_id', 'photo_cluster_label']].to_csv(os.path.join(feature_store_dir,'visual_feature_test0.csv'), sep='\t', index=False)