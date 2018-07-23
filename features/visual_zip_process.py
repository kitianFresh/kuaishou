# coding:utf8
import os
import argparse
import sys
sys.path.append('..')
import bloscpack as bp
import json
from conf.modelconf import *


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-p', '--photo_ids', help='photo_ids')
parser.add_argument('-k', '--kfold', help='the kth matrix', default=0, required=True)
args = parser.parse_args()

def data_transform_store(visual_path,matrix_store_path,photo_id_path, photo_ids):

    photo = np.load(visual_path)
    keys = photo.keys()
    prefix = keys[0].split('/')[0]
    num = len(photo_ids)
    vector = np.random.random([num, 2048])
    logging.info('Start transform single file to matrix......')
    count = 0
    for i in range(num):
        if i == 0:
            continue
        vector[i-1, :] = photo[os.path.join(prefix,photo_ids[i])]
        count += 1
        if count % 10000 == 0:
            logging.info("%s line has been processed", i)
    photo_id = np.array(photo_ids)
    bp.pack_ndarray_file(vector,matrix_store_path)
    logging.info('Matrix %s stored' % matrix_store_path)
    bp.pack_ndarray_file(photo_id,photo_id_path)
    logging.info('Matrix photo ids %s stored' % photo_id_path)



if __name__ == '__main__':
    fmt = args.format if args.format else 'csv'
    kfold = int(args.kfold)
    VISUAL_TRAIN, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'final_visual_train.zip')
    VISUAL_TEST, online_data_dir, feature_store_dir, col_feature_store_dir = get_data_file(
        'final_visual_test.zip')

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