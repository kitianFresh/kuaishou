# coding:utf8

import os
import time
import argparse
import sys

sys.path.append("..")

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



from conf.modelconf import user_action_features, face_features, user_face_favor_features, id_features, time_features, \
    photo_features, user_features, y_label, features_to_train

from common.utils import read_data, store_data
from common.base import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-v', '--version',
                    help='model version, there will be a version control and a json description file for this model',
                    required=True)
parser.add_argument('-d', '--description', help='description for a model, a json description file attached to a model',
                    required=True)

args = parser.parse_args()

if __name__ == '__main__':
    path = '../features'
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description

    model_name = 'lr-cate'
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'

    model_store_path = './sample/' if USE_SAMPLE else './data'

    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'

    model = Classifier(clf=None, dir=model_store_path,
                       name=model_name, version=version,
                       description=desc, features_to_train=features_to_train)

    CATE_TRAIN_FILE = 'ensemble_cate_feature_train'
    CATE_TRAIN_FILE = CATE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else CATE_TRAIN_FILE + '.' + fmt
    ensemble_train = read_data(os.path.join(feature_store_path, CATE_TRAIN_FILE), fmt)

    CATE_TEST_FILE = 'ensemble_cate_feature_test'
    CATE_TEST_FILE = CATE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else CATE_TEST_FILE + '.' + fmt
    ensemble_test = read_data(os.path.join(feature_store_path, CATE_TEST_FILE), fmt)


    print(ensemble_train.info())
    print(ensemble_test.info())

    all_features = list(ensemble_train.columns.values)
    print("all original features")

    print(all_features)
    y = ensemble_train[y_label].values

    features_to_train = list(set(all_features) - set(['user_id', 'photo_id', 'click', 'browse_time_diff']))
    print("train features")
    print(features_to_train)


    # ensemble_train = ensemble_train.sort_values('time')
    train_num = ensemble_train.shape[0]
    train_data = ensemble_train.iloc[:int(train_num * 0.7)].copy()
    print(train_data.shape)
    y_train = train_data[y_label].values
    val_data = ensemble_train.iloc[int(train_num * 0.7):].copy()
    print(val_data.shape)
    val_photo_ids = list(set(val_data['photo_id'].unique()) - set(train_data['photo_id'].unique()))
    val_data = val_data.loc[val_data.photo_id.isin(val_photo_ids)]
    y_val = val_data[y_label].values
    print(val_data.shape)

    num_train, num_val, num_test = train_data.shape[0], val_data.shape[0], ensemble_test.shape[0]
    ensemble_data = pd.concat([train_data[features_to_train], val_data[features_to_train], ensemble_test[features_to_train]])

    cate_feats = features_to_train
    cate_feats_indexes = [ensemble_data.columns.get_loc(c) for c in ensemble_data.columns if c in cate_feats]
    ensemble_data = ensemble_data.values
    # 1. INSTANTIATE
    enc = preprocessing.OneHotEncoder(categorical_features=cate_feats_indexes)
    # 2. FIT
    enc.fit(ensemble_data)
    # 3. Transform
    ensemble_data = enc.transform(ensemble_data)

    # pd.get_dummies must be str type not a sparse matrix by pandas get dummies
    #     ensemble_data = ensemble_data.applymap(str)
    #     ensemble_data = pd.get_dummies(ensemble_data)

    train_data = ensemble_data[:num_train, :]
    val_data = ensemble_data[num_train:(num_train+num_val), :]
    test_data = ensemble_data[(num_train+num_val):, :]

    # X = train
    # print(X.shape)
    # X_t = test
    # print(X_t.shape)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0)


    X_train, X_val, y_train, y_val = train_data, val_data, \
                                     y_train, y_val


    model.clf = LogisticRegression(C=1,verbose=True)

    start_time_1 = time.clock()
    model.clf.fit(X_train, y_train.ravel())
    print("Model trained in %s seconds" % (str(time.clock() - start_time_1)))

    model.compute_metrics(X_val, y_val.ravel())
    model.compute_features_distribution()
    model.save()
    model.submit(ensemble_test, test_data)