#coding:utf8

import os
import argparse
import sys
import time
sys.path.append("..")

import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from catboost import CatBoostClassifier

from conf.modelconf import time_features, photo_features, user_features, y_label, features_to_train
from common.utils import FeatureMerger, read_data, store_data
from common.base import Classifier

        
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-v', '--version', help='model version, there will be a version control and a json description file for this model', required=True)
parser.add_argument('-d', '--description', help='description for a model, a json description file attached to a model', required=True)
parser.add_argument('-g', '--gpu-mode', help='use gpu mode or not', action="store_true")
parser.add_argument('-a', '--all', help='use one ensemble table all, or merge by columns',action='store_true')


args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    gpu_mode = args.gpu_mode
    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description
    all_one = args.all

    
    model_name = 'catboost'

    model_store_path = './sample/' if USE_SAMPLE else './data'
    
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'

    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'

    model = Classifier(None,dir=model_store_path, name=model_name,version=version, description=desc, features_to_train=features_to_train)


    if all_one:
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
        ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.' + fmt
        ensemble_train = read_data(os.path.join(feature_store_path, ALL_FEATURE_TRAIN_FILE), fmt)

        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
        ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.' + fmt
        ensemble_test = read_data(os.path.join(feature_store_path, ALL_FEATURE_TEST_FILE), fmt)
    else:
        feature_to_use = user_features + photo_features + time_features
        fm_trainer = FeatureMerger(col_feature_store_path, feature_to_use+y_label, fmt=fmt, data_type='train', pool_type='process', num_workers=8)
        fm_tester = FeatureMerger(col_feature_store_path, feature_to_use, fmt=fmt, data_type='test', pool_type='process', num_workers=8)
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
    submission = pd.DataFrame()
    submission['user_id'] = ensemble_test['user_id']
    submission['photo_id'] = ensemble_test['photo_id']
    

    print("train features")
    print(features_to_train)    

    ensemble_offline = ensemble_train[features_to_train]
    ensemble_online = ensemble_test[features_to_train]
    # 决策树模型不需要归一化，本身就是范围划分

    print('Training model %s......' % model_name)

    ensemble_train = ensemble_train.sort_values('time')
    train_num = ensemble_train.shape[0]
    train_data = ensemble_train.iloc[:int(train_num * 0.7)].copy()
    val_data = ensemble_train.iloc[int(train_num * 0.7):].copy()

    print(train_data.shape)
    print(val_data.shape)
    val_photo_ids = list(set(val_data['photo_id'].unique()) - set(train_data['photo_id'].unique()))
    val_data = val_data.loc[val_data.photo_id.isin(val_photo_ids)]
    print(val_data.shape)
    X_train, X_val, y_train, y_val = train_data[features_to_train].values, val_data[features_to_train].values, \
                                     train_data[y_label].values, val_data[y_label].values

    print(y_train.mean(), y_train.std())
    print(y_val.mean(), y_val.std())

    start_time_1 = time.clock()
    model.clf = CatBoostClassifier(verbose=True, task_type='GPU' if gpu_mode else 'CPU')
    model.clf.fit(X_train, y_train.ravel())

    # # KFold cross validation
    # def cross_validate(*args, **kwargs):
    #     cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=False)
    #     scores = cross_val_score(model.clf, X, y.ravel(), cv=cv, scoring='roc_auc')
    #     return scores
    #
    # model.cross_validation(cross_validate)
    print("Model trained in %s seconds" % (str(time.clock() - start_time_1)))

    model.compute_metrics(X_val, y_val.ravel())
    model.compute_features_distribution()
    model.save()
    model.submit(ensemble_test)
