# coding:utf8

# must have one line in python3 between encoding and first import statement

import os
import argparse
import sys
import time

sys.path.append("../../")
from multiprocessing import cpu_count

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, RandomizedSearchCV, KFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier

from conf.modelconf import *

from common.utils import FeatureMerger, read_data, store_data, load_config_from_pyfile
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
parser.add_argument('-g', '--gpu-mode', help='use gpu mode or not', action="store_true")
parser.add_argument('-l', '--descreate-max-num', help='catboost model category feature descreate_max_num, max=48', default=30)
parser.add_argument('-b', '--bagging-num', help='catboost bagging base estimator num', default=5)




args = parser.parse_args()

if __name__ == '__main__':

    fmt = args.format if args.format else 'csv'
    gpu_mode = args.gpu_mode
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

    model_name = 'catboost-bagging'

    model_store_path = './sample/' if USE_SAMPLE else './data'

    feature_store_dir = os.path.join(online_data_dir, 'features')
    col_feature_store_dir = os.path.join(feature_store_dir, 'columns')

    model = Classifier(None, dir=model_store_path, name=model_name, version=version, description=desc,
                       features_to_train=features_to_train)

    start = time.time()
    if all_one:
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + '.' + fmt
        ensemble_train = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)

        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + '.' + fmt
        ensemble_test = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
    else:
        feature_to_use = id_features + user_features + photo_features + time_features
        fm_trainer = FeatureMerger(col_feature_store_dir, feature_to_use + y_label, fmt=fmt, data_type='train',
                                   pool_type='process', num_workers=num_workers)
        fm_tester = FeatureMerger(col_feature_store_dir, feature_to_use, fmt=fmt, data_type='test',
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

    print('开始CV %s 折训练...' % int(args.bagging_num))
    cat_feature_inds = []
    descreate_max_num = int(args.descreate_max_num)
    for i, c in enumerate(ensemble_train[features_to_train].columns):
        num_uniques = ensemble_train[features_to_train][c].nunique()
        if num_uniques < descreate_max_num:
            print(i, c, num_uniques, descreate_max_num)
            cat_feature_inds.append(i)


    class CatBoostBaggingClassfier(object):

        def __init__(self, n, kf=None):
            self.n = n
            self.kf = KFold(n_splits=n, shuffle=True, random_state=descreate_max_num * 26 + 1) if kf is None else kf
            self.cats = []

        def fit(self, X, y, cat_features=cat_feature_inds):
            t0 = time.time()
            self.cats_preds = np.zeros((X.shape[0], self.n))

            for i, (train_index, val_index) in enumerate(self.kf.split(X, y)):
                print('第{}次训练...'.format(i + 1))

                cat_model = CatBoostClassifier(
                    iterations=descreate_max_num * 50,
                    learning_rate=0.03,
                    depth=6,
                    l2_leaf_reg=1,
                    random_seed=i * 100 + 6,
                    task_type='GPU' if gpu_mode else 'CPU',
                    verbose=1,
                )
                X_train = X[train_index]
                X_val = X[val_index]
                y_train = y[train_index]
                y_val = y[val_index]
                cat_model.fit(X_train, y_train.ravel(), cat_features=cat_features, eval_set=(X_train, y_train.ravel()))
                self.cats_preds[:X_train.shape[0], i] = cat_model.predict_proba(X_train)[:, 1]
                self.cats_preds[X_train.shape[0]:, i] = cat_model.predict_proba(X_val)[:, 1]
                print('Indivisual catboost model {} train auc: {:15}'.format(i, roc_auc_score(y_train, cat_model.predict_proba(X_train)[:,1])))
                print('Indivisual catboost model {} validation auc: {:15}'.format(i, roc_auc_score(y_val, cat_model.predict_proba(X_val)[:, 1])))
                self.cats.append(cat_model)
            print('CatboostBaggingClassifier train auc: {:15}'.format(roc_auc_score(y_train, np.mean(self.cats_preds[:X_train.shape[0]],axis=1))))
            print('CatboostBaggingClassifier validation auc: {:15}'.format(roc_auc_score(y_val, np.mean(self.cats_preds[X_train.shape[0]:],axis=1))))
            print('CV训练用时{}秒'.format(time.time() - t0))

        def predict_proba(self, data, ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
            pred_probas = np.zeros((data.shape[0], 2, self.n))
            for i, cat_model in enumerate(self.cats):
                pred_probas[:,:, i] = cat_model.predict_proba(data, ntree_start, ntree_end, thread_count, verbose)
            return np.mean(pred_probas, axis=-1)

        def predict(self, data, prediction_type='Class', ntree_start=0, ntree_end=0, thread_count=-1, verbose=None):
            preds = np.zeros((data.shape[0], self.n))
            for i, cat_model in enumerate(self.cats):
                preds[:, i] = cat_model.predict(data, prediction_type, ntree_start, ntree_end, thread_count, verbose)
            from scipy import stats
            return stats.mode(preds, axis=1)[0]


    model.clf = CatBoostBaggingClassfier(int(args.bagging_num))
    model.clf.fit(ensemble_train[features_to_train].values, ensemble_train[y_label].values, cat_features=cat_feature_inds)
    # 首先声明两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，而numpy.ravel()返回的是视图（view)，会影响（reflects）原始矩阵。
    model.compute_metrics(ensemble_train[features_to_train].values, ensemble_train[y_label].values)
    model.compute_features_distribution()
    model.save()
    model.submit(ensemble_test)
