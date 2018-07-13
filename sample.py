import random
import pandas as pd
import functools
import argparse
import sys

sys.path.append("..")
from multiprocessing import cpu_count
from conf.modelconf import *


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', help='sample random seed, default 777', default=777)
parser.add_argument('-p', '--prop', help='sample propotion, default 0.15', default=0.15)
args = parser.parse_args()


random.seed(a=int(args.seed))

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

TRAIN_USER_INTERACT = os.path.join(get_root(), './data/online/train_interaction.txt')
TEST_USER_INTERACT = os.path.join(get_root(), './data/online/test_interaction.txt')
TRAIN_FACE = os.path.join(get_root(), './data/online/train_face.txt')
TEST_FACE = os.path.join(get_root(), './data/online/test_face.txt')
TRAIN_TEXT = os.path.join(get_root(), './data/online/train_text.txt')
TEST_TEXT = os.path.join(get_root(), './data/online/test_text.txt')
TRAIN_VISUAL = os.path.join(get_root(), './data/online/visual_train')
TEST_VISUAL = os.path.join(get_root(), './data/online/visual_test')



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


face_test = pd.read_csv(TEST_FACE, 
                        sep='\t', 
                        header=None, 
                        names=['photo_id', 'faces'])

user_item_test = pd.read_csv(TEST_USER_INTERACT, 
                             sep='\t', 
                             header=None, 
                             names=['user_id', 'photo_id', 'time', 'duration_time'])

text_test = pd.read_csv(TEST_TEXT,
                       sep='\t',
                       header=None,
                       names=['photo_id', 'cover_words'])

user_sample = functools.partial(user_sample, prop=float(args.prop))
sample_user_item_test = user_item_test.groupby(['user_id']).apply(user_sample)
sample_user_item_test.reset_index(drop=True, inplace=True)
sample_user_item_test.to_csv('./sample/online/test_interaction.txt', sep='\t', index=False, header=False)
print('sample_user_item_test sampled %d from %d user_item_test data' % (sample_user_item_test.shape[0], user_item_test.shape[0]))

user_sample = functools.partial(user_sample, prop=float(args.prop))
sample_user_item_train = user_item_train.groupby(['user_id']).apply(user_sample)
sample_user_item_train.reset_index(drop=True, inplace=True)
sample_user_item_train.to_csv('./sample/online/train_interaction.txt', sep='\t', index=False, header=False)
print('sample_user_item_train sampled %d from %d user_item_train data' % (sample_user_item_train.shape[0], user_item_train.shape[0]))

sample_photo_ids_test = set(sample_user_item_test['photo_id'].unique())
sample_photo_ids_train = set(sample_user_item_train['photo_id'].unique())

sample_face_train = photo_sample(face_train, sample_photo_ids_train)
sample_face_train.to_csv('./sample/online/train_face.txt', sep='\t', index=False, header=False)
print('sample_face_train sampled %d from %d face_train data' % (sample_face_train.shape[0], face_train.shape[0]))


sample_face_test = photo_sample(face_test, sample_photo_ids_test)
sample_face_test.to_csv('./sample/online/test_face.txt', sep='\t', index=False, header=False)
print('sample_face_test sampled %d from %d face_test data' % (sample_face_test.shape[0], face_test.shape[0]))


sample_text_test = photo_sample(text_test, sample_photo_ids_test)
sample_text_test.to_csv('./sample/online/test_text.txt', sep='\t', index=False, header=False)
print('sample_text_test sampled %d from %d text_test data' % (sample_text_test.shape[0], text_test.shape[0]))


sample_text_train = photo_sample(text_train, sample_photo_ids_train)
sample_text_train.to_csv('./sample/online/train_text.txt', sep='\t', index=False, header=False)
print('sample_text_train sampled %d from %d text_train data' % (sample_text_train.shape[0], text_train.shape[0]))

# visual_train_path = './data/visual_train'
# visual_test_path = './data/visual_test'
#
# sample_visual_train_path = './sample/visual_train'
# sample_visual_test_path = './sample/visual_test'
#
# sample_visual_train = open(sample_visual_train_path,'w')
# for photo_id in sample_photo_ids_train:
#     sample_visual_train.write(visual_train_path + '/' + str(photo_id))
#     sample_visual_train.write('\n')
# sample_visual_train.close()
# print('visual_train sampled')
#
# sample_visual_test = open(sample_visual_test_path,'w')
# for photo_id in sample_photo_ids_test:
#     sample_visual_test.write(visual_test_path + '/' + str(photo_id))
#     sample_visual_test.write('\n')
# sample_visual_test.close()
# print('visual_test sampled')