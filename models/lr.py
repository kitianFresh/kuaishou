#coding:utf8

import os
import argparse
import time
import sys
sys.path.append("..")

import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from conf.modelconf import user_action_features, face_features, user_face_favor_features, id_features, time_features, photo_features, user_features, y_label, features_to_train, norm_features

from common.utils import read_data, store_data, normalize_min_max, normalize_z_score
from common.base import Classifier

        
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-v', '--version', help='model version, there will be a version control and a json description file for this model', required=True)
parser.add_argument('-d', '--description', help='description for a model, a json description file attached to a model', required=True)

args = parser.parse_args()


if __name__ == '__main__':
    
    path = '../features'
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description
    
    model_name = 'lr'

    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'

    model_store_path = './sample/' if USE_SAMPLE else './data'

    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'

    model = Classifier(clf=None, dir=model_store_path,
                       name=model_name, version=version,
                       description=desc, features_to_train=features_to_train)

    start = time.clock()
    ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
    ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.' + fmt
    ensemble_train = read_data(os.path.join(feature_store_path, ALL_FEATURE_TRAIN_FILE), fmt)

    ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
    ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.' + fmt
    ensemble_test = read_data(os.path.join(feature_store_path, ALL_FEATURE_TEST_FILE), fmt)
    print('Reading data in %s seconds' % str(time.clock()-start))
    
    print(ensemble_train.info())
    print(ensemble_test.info())

    all_features = list(ensemble_train.columns.values)
    print("all original features")
    print(all_features) 
    y = ensemble_train[y_label].values
    
    # less features to avoid overfit

    print("train features")
    print(features_to_train)
    
    num_train, num_test = ensemble_train.shape[0], ensemble_test.shape[0]
    ensemble_data = pd.concat([ensemble_train[id_features+features_to_train], ensemble_test[id_features+features_to_train]])
    normalize_min_max(ensemble_data, norm_features)
    train = ensemble_data.iloc[:num_train, :]
    train = pd.concat([train, ensemble_train[y_label]], axis=1)
    test = ensemble_data.iloc[num_train:,:]


    # X = train[features_to_train].values
    # print(X.shape)
    # X_t = test[features_to_train].values
    # print(X_t.shape)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)


    train = train.sort_values('time')
    train_num = train.shape[0]
    train_data = train.iloc[:int(train_num * 0.7)].copy()
    print(train_data.shape)
    val_data = train.iloc[int(train_num * 0.7):].copy()
    print(val_data.shape)
    val_photo_ids = list(set(val_data['photo_id'].unique()) - set(train_data['photo_id'].unique()))
    val_data = val_data.loc[val_data.photo_id.isin(val_photo_ids)]
    print(val_data.shape)
    X_train, X_val, y_train, y_val = train_data[features_to_train].values, val_data[features_to_train].values, \
                                     train_data[y_label].values, val_data[y_label].values

    model.clf = LogisticRegression(C=1, verbose=True)

    start_time_1 = time.clock()
    model.clf.fit(X_train, y_train.ravel())
    print("Model trained in %s seconds" % (str(time.clock() - start_time_1)))

    model.compute_metrics(X_val, y_val.ravel())
    model.compute_features_distribution()
    model.save()
    model.submit(ensemble_test)