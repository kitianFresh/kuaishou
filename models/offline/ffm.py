import xlearn as xl

# coding:utf8

import os
import argparse
import sys
import time

sys.path.append("../../")
from multiprocessing import cpu_count

import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

from conf.modelconf import *
from common.utils import FeatureMerger, read_data, store_data
from common.base import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-v', '--version',
                    help='model version, there will be a version control and a json description file for this model',
                    required=True)
parser.add_argument('-d', '--description', help='description for a model, a json description file attached to a model',
                    required=True)
parser.add_argument('-a', '--all', help='use one ensemble table all, or merge by columns', action='store_true')
parser.add_argument('-n', '--num-workers', help='num used to merge columns', default=cpu_count())
parser.add_argument('-c', '--config-file', help='model config file', default='')

args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description
    all_one = args.all
    num_workers = args.num_workers
    kfold = 0
    model_name = 'ffm'

    model_store_path = './sample/' if USE_SAMPLE else './data'

    feature_store_dir = os.path.join(offline_data_dir, 'feature')
    col_feature_store_dir = os.path.join(feature_store_dir, 'columns')

    model = Classifier(None, dir=model_store_path, name=model_name, version=version, description=desc,
                       features_to_train=features_to_train)

    ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + str(kfold) + '.' + fmt

    ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + str(kfold) + '.' + fmt



    # less features to avoid overfit
    # features_to_train = ['exposure_num', 'click_ratio', 'cover_length_favor', 'woman_yen_value_favor', 'woman_cv_favor', 'cover_length', 'browse_num', 'man_age_favor', 'woman_age_favor', 'time', 'woman_scale', 'duration_time', 'woman_favor', 'playing_ratio', 'face_click_favor', 'click_num', 'man_cv_favor', 'man_scale', 'playing_sum', 'man_yen_value_favor', 'man_avg_age', 'playing_freq', 'woman_avg_attr', 'human_scale', 'browse_freq', 'non_face_click_favor', 'click_freq', 'woman_avg_age', 'human_avg_attr', 'duration_sum', 'man_favor', 'human_avg_age', 'follow_ratio', 'man_avg_attr']

    print("train features")
    print(features_to_train)

    print('Training model %s......' % model_name)

    import xlearn as xl

    # Training task
    ffm_model = xl.create_ffm()  # Use field-aware factorization machine
    ffm_model.setTrain(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE))  # Training data
    ffm_model.setValidate(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE))  # Validation data

    # param:
    #  0. binary classification
    #  1. learning rate: 0.2
    #  2. regular lambda: 0.002
    #  3. evaluation metric: accuracy
    param = {'task': 'binary', 'lr': 0.2,
             'lambda': 0.002, 'metric': 'auc'}

    # Start to train
    # The trained model will be stored in model.out

    # Prediction task
    # ffm_model.setTest("./small_test.txt")  # Test data
    # ffm_model.setSigmoid()  # Convert output to 0-1

    # Start to predict
    # The output result will be stored in output.txt
    # ffm_model.predict("./model.out", "./output.txt")

    model.clf = ffm_model
    model.clf.fit(param, os.path.join(model_store_path, 'ffm/ffm.model'))

    model.compute_features_distribution()

    model.save()


