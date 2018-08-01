# coding:utf8

# must have one line in python3 between encoding and first import statement

import os
import argparse
import sys
import time

sys.path.append("../../")
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf

from common.utils import FeatureMerger, read_data, store_data, load_config_from_pyfile
from common.base import Classifier
from conf.modelconf import *

from common.deep_model import DCN

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
parser.add_argument('-g', '--gpu-mode', help='use gpu mode or not', action="store_true")

args = parser.parse_args()

class FeatureDictionary(object):
    def __init__(self, trainfile=None,testfile=None,
                 numeric_cols=[],
                 ignore_cols=[],
                 cate_cols=[]):

        self.trainfile = trainfile
        #self.testfile = testfile
        self.testfile = testfile
        self.cate_cols = cate_cols
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        df = pd.concat([self.trainfile,self.testfile])
        self.feat_dict = {}
        self.feat_len = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols or col in self.numeric_cols:
                continue
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        self.feat_dim = tc




class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict


    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            y = dfi["click"].values.tolist()
            dfi.drop(["user_id", "photo_id", "click"], axis=1, inplace=True)
        else:
            ids = dfi[["user_id", "photo_id"]]
            dfi.drop(["user_id", "photo_id"], axis=1, inplace=True)
            dfi.drop(["click"], axis=1, inplace=True)

        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)

        numeric_Xv = dfi[self.feat_dict.numeric_cols].values.tolist()
        dfi.drop(self.feat_dict.numeric_cols,axis=1,inplace=True)

        dfv = dfi.copy()
        cate_used_cols = []
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            else:
                cate_used_cols.append(col)
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        cate_Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        cate_Xv = dfv.values.tolist()
        print(len(cate_Xi), len(cate_Xi[0]))
        print(len(cate_Xv), len(cate_Xv[0]))
        print(cate_used_cols, len(cate_used_cols))
        print(cate_Xi[0])
        print(cate_Xv[0])
        if has_label:
            return cate_Xi, cate_Xv,numeric_Xv,y
        else:
            return cate_Xi, cate_Xv,numeric_Xv,ids


if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description
    all_one = args.all
    kfold = 0
    num_workers = args.num_workers
    config = load_config_from_pyfile(args.config_file)
    features_to_train = config.features_to_train
    seed = config.seed
    y_label = config.y_label

    logging.info('--------------------version: %s ----------------------' % version)
    logging.info('desc: %s' % desc)
    model_name = 'dcn'

    model_store_path = './sample/' if USE_SAMPLE else './data'

    feature_store_dir = os.path.join(offline_data_dir, 'features')
    col_feature_store_dir = os.path.join(feature_store_dir, 'columns')

    model = Classifier(None, dir=model_store_path, name=model_name, version=version, description=desc,
                       features_to_train=features_to_train)

    start = time.time()
    cover_words_vector_train = sparse.load_npz(
        os.path.join(feature_store_dir, 'cover_words_vector_feature_train' + str(kfold) + '.npz'))
    cover_words_vector_test = sparse.load_npz(
        os.path.join(feature_store_dir, 'cover_words_vector_feature_test' + str(kfold) + '.npz'))

    if all_one:
        ALL_FEATURE_TRAIN_FILE = 'deep_feature_train' + str(kfold) + '.' + fmt
        ensemble_train = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)

        ALL_FEATURE_TEST_FILE = 'deep_feature_test' + str(kfold) + '.' + fmt
        ensemble_test = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
    else:
        deep_col_feature_store_dir = os.path.join(col_feature_store_dir, "deep")
        feature_to_use = id_features + features_to_train
        fm_trainer = FeatureMerger(deep_col_feature_store_dir, feature_to_use + y_label, fmt=fmt, data_type='train',
                                   pool_type='process', num_workers=num_workers)
        fm_tester = FeatureMerger(deep_col_feature_store_dir, feature_to_use + y_label, fmt=fmt, data_type='test',
                                  pool_type='process', num_workers=num_workers)
        ensemble_train = fm_trainer.concat()
        ensemble_test = fm_tester.concat()

    end = time.time()
    print('data read in %s seconds' % str(end - start))

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

    print(ensemble_train[y_label].mean(), ensemble_train[y_label].std())
    print(ensemble_test[y_label].mean(), ensemble_test[y_label].std())
    print(ensemble_train[features_to_train].shape, ensemble_train[y_label].shape)
    print(ensemble_test[features_to_train].shape, ensemble_test[features_to_train].shape)


    import gc
    gc.collect()
    start_time_1 = time.time()

    dcn_params = {

        "embedding_size": 8,
        "deep_layers": [32, 32],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "random_seed": seed,
        "cross_layer_num": 3,
    }


    fd = FeatureDictionary(ensemble_train, ensemble_test, numeric_cols=config.numeric_features,
                           ignore_cols=config.ignore_features,
                           cate_cols=config.cate_features)

    # print(fd.feat_dim)
    # print(fd.feat_dict)


    data_parser = DataParser(feat_dict=fd)
    cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train = data_parser.parse(df=ensemble_train, has_label=True)
    cate_Xi_valid, cate_Xi_valid, numeric_Xv_valid, y_valid = data_parser.parse(df=ensemble_test, has_label=True)

    # cate_Xi_valid, cate_Xi_valid, numeric_Xv_valid, ids_test = data_parser.parse(df=ensemble_test)

    dcn_params["cate_feature_size"] = fd.feat_dim
    dcn_params["field_size"] = len(cate_Xi_train[0])
    dcn_params['numeric_feature_size'] = len(config.numeric_features)

    dcn = DCN(**dcn_params)

    dcn.fit(cate_Xi_train, cate_Xv_train, numeric_Xv_train, y_train, cate_Xi_valid, cate_Xi_valid,
                numeric_Xv_valid, y_valid)


