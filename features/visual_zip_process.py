# coding:utf8
import os
import argparse
import sys
sys.path.append('..')
import bloscpack as bp
import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

from conf.modelconf import *


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-p', '--photo_ids', help='photo_ids')
parser.add_argument('-k', '--kfold', help='the kth matrix', default=0, required=True)
args = parser.parse_args()


def matrix_fill(visual_dir, photo_id, matrix, i):
    path = os.path.join(visual_dir, str(photo_id))
    photo = np.load(path)
    matrix[i, :] = photo
    return i, photo_id

def data_transform_store(visual_dir,matrix_store_path,photo_id_path, photo_ids, workers=cpu_count()):

    num = len(photo_ids)
    matrix = np.random.random([num, 2048])
    logging.info('Start transform single file to matrix......')
    count = 0
    tasks_args = []
    for i, photo_id in enumerate(photo_ids):
        tasks_args.append((visual_dir, photo_id, matrix, i))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_results = (res for res in executor.map(matrix_fill, tasks_args))
    for i, photo_id in future_results:
        count += 1
        if count % 10000 == 0:
            logging.info("%s photos has been processed", i)
    photo_id = np.array(photo_ids)
    bp.pack_ndarray_file(matrix, matrix_store_path)
    logging.info('Matrix %s stored' % matrix_store_path)
    bp.pack_ndarray_file(photo_id,photo_id_path)
    logging.info('Matrix photo ids %s stored' % photo_id_path)



if __name__ == '__main__':
    fmt = args.format if args.format else 'csv'
    kfold = int(args.kfold)
    VISUAL_TRAIN, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'visual_train')
    VISUAL_TEST, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'visual_test')

    TRAIN_VISUAL_MATRIX = os.path.join(online_data_dir,'visual_train_matrix' + str(kfold) + '.blp')
    TEST_VISUAL_MATRIX = os.path.join(online_data_dir,  'visual_test_matrix' + str(kfold) + '.blp')
    TRAIN_VISUAL_PHOTO_ID = os.path.join(online_data_dir, 'visual_train_photo_id' + str(kfold) + '.blp')
    TEST_VISUAL_PHOTO_ID = os.path.join(online_data_dir, 'visual_test_photo_id' + str(kfold) + '.blp')
    path = os.path.join(online_data_dir, 'photo_ids.json')
    with open(path, 'r') as f:
        data = json.load(f)
    photo_ids_train = data['photo_ids_train' + str(kfold)]
    photo_ids_test  = data['photo_ids_test' + str(kfold)]
    data_transform_store(VISUAL_TRAIN, TRAIN_VISUAL_MATRIX, TRAIN_VISUAL_PHOTO_ID, photo_ids_train)
    data_transform_store(VISUAL_TEST, TEST_VISUAL_MATRIX, TEST_VISUAL_PHOTO_ID, photo_ids_test)