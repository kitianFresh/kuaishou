# coding:utf8
import os
import argparse
import sys
sys.path.append('..')
import bloscpack as bp
import gc
import json
from multiprocessing import cpu_count
from conf.modelconf import *


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-k', '--kfold', help='k matrix', default=cpu_count(), required=True)
args = parser.parse_args()



if __name__ == '__main__':
    fmt = args.format if args.format else 'csv'
    kfold = int(args.kfold)
    VISUAL_TRAIN, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'final_visual_train.zip')
    VISUAL_TEST, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'final_visual_test.zip')

    TRAIN_VISUAL_MATRIX = os.path.join(online_data_dir, 'visual_train_matrix.blp')
    TEST_VISUAL_MATRIX = os.path.join(online_data_dir, 'visual_test_matrix.blp')
    TRAIN_VISUAL_PHOTO_ID = os.path.join(online_data_dir, 'visual_train_photo_id.blp')
    TEST_VISUAL_PHOTO_ID = os.path.join(online_data_dir, 'visual_test_photo_id.blp')

    path = os.path.join(online_data_dir, 'photo_ids.json')
    with open(path, 'r') as f:
        data = json.load(f)

    sum_train = 0
    sum_test = 0
    for k in range(kfold):
        photo_ids_train = data['photo_ids_train' + str(k)]
        photo_ids_test = data['photo_ids_test' + str(k)]
        sum_train += len(photo_ids_train)
        sum_test += len(photo_ids_test)

    vector_train = np.random.random([sum_train, 2048])
    vector_test = np.random.random([sum_test, 2048])

    photo_ids_train = np.arange(sum_train)
    photo_ids_test = np.arange(sum_test)
    num_train = 0
    num_test = 0
    for k in range(kfold):
        TRAIN_VISUAL_MATRIX_K = os.path.join(online_data_dir,'visual_train_matrix' + str(k) + '.blp')
        TEST_VISUAL_MATRIX_K = os.path.join(online_data_dir,  'visual_test_matrix' + str(k) + '.blp')
        TRAIN_VISUAL_PHOTO_ID_K = os.path.join(online_data_dir, 'visual_train_photo_id' + str(k) + '.blp')
        TEST_VISUAL_PHOTO_ID_K = os.path.join(online_data_dir, 'visual_test_photo_id' + str(k) + '.blp')

        vec_train_k = bp.unpack_ndarray_file(TRAIN_VISUAL_MATRIX_K)
        vec_test_k = bp.unpack_ndarray_file(TEST_VISUAL_MATRIX_K)
        photo_ids_train_k = bp.unpack_ndarray_file(TRAIN_VISUAL_PHOTO_ID_K)
        photo_ids_test_k = bp.unpack_ndarray_file(TEST_VISUAL_PHOTO_ID_K)
        for i, vec in enumerate(vec_train_k):
            vector_train[num_train+i, :] = vec
            photo_ids_train[num_train+i] = photo_ids_train_k
        for i, vec in enumerate(vec_test_k):
            vector_test[num_test+i, :] = vec
            photo_ids_test[num_test+i] = photo_ids_test_k
        num_train += vec_train_k.shape[0]
        num_test += vec_test_k.shape[0]
        del vec_test_k
        del vec_train_k
        del photo_ids_train_k
        del photo_ids_test_k
        gc.collect()
    bp.pack_ndarray_file(vector_train, TRAIN_VISUAL_MATRIX)
    bp.pack_ndarray_file(vector_test, TEST_VISUAL_MATRIX)
    bp.pack_ndarray_file(photo_ids_train, TRAIN_VISUAL_PHOTO_ID)
    bp.pack_ndarray_file(photo_ids_test, TEST_VISUAL_PHOTO_ID)