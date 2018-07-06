import random
import pandas as pd
import numpy as np
import functools

random.seed(a=777)

def user_sample(group, prop):
    n = group.shape[0]
    m = int(prop * n)
    m = n if m == 0 else m
    group = group.iloc[random.sample(range(n), m)]
    return group

def photo_sample(df, photo_ids):
    t = df[df['photo_id'].isin(photo_ids)]
    return t

face_train = pd.read_csv('./data/train_face.txt', 
                        sep='\t', 
                        header=None, 
                        names=['photo_id', 'faces'])

user_item_train = pd.read_csv('./data/train_interaction.txt', 
                             sep='\t', 
                             header=None, 
                             names=['user_id', 'photo_id', 'click', 'like', 'follow', 'time', 'playing_time', 'duration_time'])

text_train = pd.read_csv('./data/train_text.txt',
                       sep='\t',
                       header=None,
                       names=['photo_id', 'cover_words'])

face_test = pd.read_csv('./data/test_face.txt', 
                        sep='\t', 
                        header=None, 
                        names=['photo_id', 'faces'])

user_item_test = pd.read_csv('./data/test_interaction.txt', 
                             sep='\t', 
                             header=None, 
                             names=['user_id', 'photo_id', 'time', 'duration_time'])

text_test = pd.read_csv('./data/test_text.txt',
                       sep='\t',
                       header=None,
                       names=['photo_id', 'cover_words'])

user_sample = functools.partial(user_sample, prop=0.15)
sample_user_item_test = user_item_test.groupby(['user_id']).apply(user_sample)
sample_user_item_test.reset_index(drop=True, inplace=True)
sample_user_item_test.to_csv('./sample/test_interaction.txt', sep='\t', index=False, header=False)
print('user_item_test sampled')

user_sample = functools.partial(user_sample, prop=0.15)
sample_user_item_train = user_item_train.groupby(['user_id']).apply(user_sample)
sample_user_item_train.reset_index(drop=True, inplace=True)
sample_user_item_train.to_csv('./sample/train_interaction.txt', sep='\t', index=False, header=False)
print('user_item_train sampled')

sample_photo_ids_test = set(sample_user_item_test['photo_id'].unique())
sample_photo_ids_train = set(sample_user_item_train['photo_id'].unique())

sample_face_train = photo_sample(face_train, sample_photo_ids_train)
sample_face_train.to_csv('./sample/train_face.txt', sep='\t', index=False, header=False)
print('face_train sampled')


sample_face_test = photo_sample(face_test, sample_photo_ids_test)
sample_face_test.to_csv('./sample/test_face.txt', sep='\t', index=False, header=False)
print('face_test sampled')


sample_text_test = photo_sample(text_test, sample_photo_ids_test)
sample_text_test.to_csv('./sample/test_text.txt', sep='\t', index=False, header=False)
print('text_test sampled')


sample_text_train = photo_sample(text_train, sample_photo_ids_train)
sample_text_train.to_csv('./sample/train_text.txt', sep='\t', index=False, header=False)
print('text_train sampled')

visual_train_path = './data/visual_train'
visual_test_path = './data/visual_test'

sample_visual_train_path = './sample/visual_train'
sample_visual_test_path = './sample/visual_test'

sample_visual_train = open(sample_visual_train_path,'w')
for photo_id in sample_photo_ids_train:
    sample_visual_train.write(visual_train_path + '/' + str(photo_id))
    sample_visual_train.write('\n')
sample_visual_train.close()
print('visual_train sampled')

sample_visual_test = open(sample_visual_test_path,'w')
for photo_id in sample_photo_ids_test:
    sample_visual_test.write(visual_test_path + '/' + str(photo_id))
    sample_visual_test.write('\n')
sample_visual_test.close()
print('visual_test sampled')