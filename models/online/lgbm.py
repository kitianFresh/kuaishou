#coding:utf8

# must have one line in python3 between encoding and first import statement

import os
import argparse
import sys
import time
sys.path.append("../../")
from multiprocessing import cpu_count

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, RandomizedSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV

from lightgbm import LGBMClassifier

from common.utils import FeatureMerger, read_data, store_data, load_config_from_pyfile
from common.base import Classifier
from conf.modelconf import *

        
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-v', '--version', help='model version, there will be a version control and a json description file for this model', required=True)
parser.add_argument('-d', '--description', help='description for a model, a json description file attached to a model', required=True)
parser.add_argument('-a', '--all', help='use one ensemble table all, or merge by columns',action='store_true')
parser.add_argument('-n', '--num-workers', help='num used to merge columns', default=cpu_count())
parser.add_argument('-c', '--config-file', help='model config file', default='')
parser.add_argument('-g', '--gpu-mode', help='use gpu mode or not', action="store_true")

args = parser.parse_args()


if __name__ == '__main__':
    
    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description
    all_one = args.all
    num_workers = args.num_workers
    config = load_config_from_pyfile(args.config_file)
    features_to_train = config.features_to_train
    user_features = config.user_features
    photo_features = config.photo_features
    time_features = config.time_features
    y_label = config.y_label
    
    model_name = 'lgbm'

    model_store_path = './sample/' if USE_SAMPLE else './data'

    feature_store_dir = os.path.join(online_data_dir, 'features')
    col_feature_store_dir = os.path.join(feature_store_dir, 'columns')

    model = Classifier(None,dir=model_store_path, name=model_name,version=version, description=desc, features_to_train=features_to_train)


    if all_one:
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + '.' + fmt
        ensemble_train = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)

        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + '.' + fmt
        ensemble_test = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
    else:
        feature_to_use = user_features + photo_features + time_features
        fm_trainer = FeatureMerger(col_feature_store_dir, feature_to_use+y_label, fmt=fmt, data_type='train', pool_type='process', num_workers=num_workers)
        fm_tester = FeatureMerger(col_feature_store_dir, feature_to_use, fmt=fmt, data_type='test', pool_type='process', num_workers=num_workers)
        ensemble_train = fm_trainer.merge()
        ensemble_test = fm_tester.merge()

    print(ensemble_train.info())
    print(ensemble_test.info())

    all_features = list(ensemble_train.columns.values)
    print("all original features")
    print(all_features) 
    y = ensemble_train[y_label].values
    
    # less features to avoid overfit
    # features_to_train = ['exposure_num', 'click_ratio', 'cover_length_favor', 'woman_yen_value_favor', 'woman_cv_favor', 'cover_length', 'browse_num', 'man_age_favor', 'woman_age_favor', 'time', 'woman_scale', 'duration_time', 'woman_favor', 'playing_ratio', 'face_click_favor', 'click_num', 'man_cv_favor', 'man_scale', 'playing_sum', 'man_yen_value_favor', 'man_avg_age', 'playing_freq', 'woman_avg_attr', 'human_scale', 'browse_freq', 'non_face_click_favor', 'click_freq', 'woman_avg_age', 'human_avg_attr', 'duration_sum', 'man_favor', 'human_avg_age', 'follow_ratio', 'man_avg_attr']

    print("train features")
    print(features_to_train)

    # from sklearn.utils import shuffle
    # ensemble_train = shuffle(ensemble_train)
    # 这样划分出来的数据，训练集和验证集点击率分布不台符合
    # train_data, val_data, y_train, y_val = train_test_split(ensemble_train[id_features+features_to_train+y_label], ensemble_train[y_label], test_size=0.3, random_state=0)

    print('Training model %s......' % model_name)

    X_train, y_train = ensemble_train[features_to_train].values, \
                                     ensemble_train[y_label].values

    print(y_train.mean(), y_train.std())
    import gc
    del ensemble_train
    gc.collect()
    start_time_1 = time.clock()

    model.clf = LGBMClassifier(boosting_type='gbdt', num_leaves=127,
                                max_depth=5, learning_rate=0.01,
                                n_estimators=500,objective='binary',
                                min_split_gain=0.0, min_child_weight=0.001,
                                min_child_samples=200, subsample=0.8,
                                subsample_freq=0, colsample_bytree=0.8,
                                reg_alpha=0.0, reg_lambda=0.0,
                                random_state=2018, n_jobs=-1, device='gpu' if args.gpu_mode else 'cpu',
                                silent=False)
    model.clf.fit(X_train, y_train.ravel(),eval_set=[(X_train, y_train.ravel())], eval_metric='auc')
    print("Model trained in %s seconds" % (str(time.clock() - start_time_1)))

    # 首先声明两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，而numpy.ravel()返回的是视图（view)，会影响（reflects）原始矩阵。
    model.compute_metrics(X_train, y_train.ravel())
    model.compute_features_distribution()
    model.save()
    model.submit(ensemble_test)
