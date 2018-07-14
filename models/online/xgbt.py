# coding:utf8

# must have one line in python3 between encoding and first import statement

import os
import argparse
import sys
import time

sys.path.append("..")
from multiprocessing import cpu_count

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, RandomizedSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV

from xgboost import XGBClassifier
# from conf.modelconf import user_action_features, face_features, user_face_favor_features, id_features, time_features, photo_features, user_features, y_label, features_to_train

from common.utils import FeatureMerger, read_data, store_data, load_config_from_pyfile
from common.base import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
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

    USE_SAMPLE = args.sample
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

    model_name = 'xgbt'

    model_store_path = './sample/' if USE_SAMPLE else './data'

    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'

    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'

    model = Classifier(None, dir=model_store_path, name=model_name, version=version, description=desc,
                       features_to_train=features_to_train)

    if all_one:
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
        ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.' + fmt
        ensemble_train = read_data(os.path.join(feature_store_path, ALL_FEATURE_TRAIN_FILE), fmt)

        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
        ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.' + fmt
        ensemble_test = read_data(os.path.join(feature_store_path, ALL_FEATURE_TEST_FILE), fmt)
    else:
        feature_to_use = user_features + photo_features + time_features
        fm_trainer = FeatureMerger(col_feature_store_path, feature_to_use + y_label, fmt=fmt, data_type='train',
                                   pool_type='process', num_workers=num_workers)
        fm_tester = FeatureMerger(col_feature_store_path, feature_to_use, fmt=fmt, data_type='test',
                                  pool_type='process', num_workers=num_workers)
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

    # RandomizedSearchCV参数说明，clf1设置训练的学习器
    # param_dist字典类型，放入参数搜索范围
    # scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
    # n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
    # n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU
    # model.clf = RandomizedSearchCV(LGBMClassifier(), param_dist, cv=3, scoring='roc_auc', n_iter=300, n_jobs=-1)

    ind_params = {
        'random_state': 32,
        'objective': 'binary:logistic',
        'n_estimators': 200,
        'learning_rate': 0.1,
    }
    params = {'max_depth': (4, 6, 8),
              'subsample': (0.75, 0.8, 0.9, 1.0),
              'colsample_bytree': (0.75, 0.8, 0.9, 1.0),
              'gamma': [i / 10 for i in range(0, 5)]
              }

    clf2 = EvolutionaryAlgorithmSearchCV(estimator=XGBClassifier(**ind_params),
                                         params=params,
                                         scoring="roc_auc",
                                         cv=3,
                                         verbose=1,
                                         population_size=60,
                                         gene_mutation_prob=0.10,
                                         gene_crossover_prob=0.5,
                                         tournament_size=5,
                                         generations_number=100,
                                         n_jobs=8)
    # 在训练集上训练
    model.clf.fit(X_train, y_train.ravel())

    # KFold cross validation
    # def cross_validate(*args, **kwargs):
    #     cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=False)
    #     scores = cross_val_score(model.clf, X, y.ravel(), cv=cv, scoring='roc_auc')
    #     return scores
    # model.cross_validation(cross_validate)
    print("Model trained in %s seconds" % (str(time.clock() - start_time_1)))

    # 首先声明两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，而numpy.ravel()返回的是视图（view)，会影响（reflects）原始矩阵。
    model.compute_metrics(X_val, y_val.ravel())
    model.compute_features_distribution()
    model.save()
    model.submit(ensemble_test)