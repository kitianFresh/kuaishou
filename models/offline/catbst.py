#coding:utf8

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
parser.add_argument('-v', '--version', help='model version, there will be a version control and a json description file for this model', required=True)
parser.add_argument('-d', '--description', help='description for a model, a json description file attached to a model', required=True)
parser.add_argument('-g', '--gpu-mode', help='use gpu mode or not', action="store_true")
parser.add_argument('-a', '--all', help='use one ensemble table all, or merge by columns',action='store_true')
parser.add_argument('-r', '--down-sampling', help='down sampling rate, default 1. no sampling', default=1.)
parser.add_argument('-k', '--topk-features', help='top k features to use again to train model', default=100)
parser.add_argument('-n', '--num-workers', help='num used to merge columns', default=cpu_count())
parser.add_argument('-c', '--config-file', help='model config file', default='')
parser.add_argument('-l', '--descreate-max-num', help='catboost model category feature descreate_max_num, max=48', default=30)



args = parser.parse_args()

if __name__ == '__main__':
    
    gpu_mode = args.gpu_mode
    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description
    all_one = args.all
    down_sample_rate = float(args.down_sampling)
    k = int(args.topk_features)
    num_workers = args.num_workers
    kfold = 0
    model_name = 'catboost'

    model_store_path = './sample/' if USE_SAMPLE else './data'

    feature_store_dir = os.path.join(offline_data_dir, 'features')
    col_feature_store_dir = os.path.join(feature_store_dir, 'columns')

    model = Classifier(None,dir=model_store_path, name=model_name,version=version, description=desc, features_to_train=features_to_train)


    start = time.time()
    if all_one:
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + str(kfold) + '.' + fmt
        ensemble_train = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)

        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + str(kfold) + '.' + fmt
        ensemble_test = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
    else:
        feature_to_use = user_features + photo_features + time_features
        fm_trainer = FeatureMerger(col_feature_store_dir, feature_to_use+y_label, fmt=fmt, data_type='train', pool_type='process', num_workers=num_workers)
        fm_tester = FeatureMerger(col_feature_store_dir, feature_to_use, fmt=fmt, data_type='test', pool_type='process', num_workers=num_workers)
        ensemble_train = fm_trainer.merge()
        ensemble_test = fm_tester.merge()

    end = time.time()
    print('data read in %s seconds' % str(end-start))

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
    if down_sample_rate < 1.:
        print('down sampling by %f' % down_sample_rate)
        ensemble_train_neg = ensemble_train[ensemble_train[y_label[0]] == 0]
        print(ensemble_train_neg.shape)
        ensemble_train_pos = ensemble_train[ensemble_train[y_label[0]] == 1]
        print(ensemble_train_pos.shape)
        ensemble_train_neg = ensemble_train_neg.sample(frac=down_sample_rate)
        print(ensemble_train_neg.shape)
        ensemble_train = pd.concat([ensemble_train_pos, ensemble_train_neg])

    print('Training model %s......' % model_name)

    X_train, y_train = ensemble_train[features_to_train].values, ensemble_train[y_label].values

    X_val, y_val = ensemble_test[features_to_train].values, ensemble_test[y_label].values
    
    print(y_train.mean(), y_train.std())
    print(y_val.mean(), y_val.std())


    cat_feature_inds = []
    descreate_max_num = int(args.descreate_max_num)
    descreate_max_num = 48 if descreate_max_num > 48 else descreate_max_num
    for i, c in enumerate(ensemble_train[features_to_train].columns):
        num_uniques = ensemble_train[features_to_train][c].nunique()
        if num_uniques < descreate_max_num:
            print(c, num_uniques, descreate_max_num)
            cat_feature_inds.append(i)
    start = time.time()
    model.clf = CatBoostClassifier(iterations=1500,
                                   task_type='GPU' if gpu_mode else 'CPU',
                                   gpu_cat_features_storage='CpuPinnedMemory',
                                   pinned_memory_size=1073741824 * 8,
                                   used_ram_limit='200gb',
                                   boosting_type="Plain",
                                   simple_ctr='Borderes:PriorEstimation=BetaPrior:Prior=0.2',
                                   save_snapshot=True,
                                   snapshot_file=os.path.join(model_store_path, model_name + '-' + version + '.bak'),
                                   logging_level='Debug')

    # model.clf = CatBoostClassifier(iterations=3000, task_type='GPU' if gpu_mode else 'CPU', gpu_cat_features_storage='CpuPinnedMemory', pinned_memory_size=1073741824*8, used_ram_limit='200gb', save_snapshot=True,snapshot_file=os.path.join(model_store_path, model_name + '-' + version + '.bak'))

    model.clf.fit(X_train, y_train.ravel(), cat_features=cat_feature_inds, eval_set=(X_val, y_val.ravel()))
    
    print("Model trained in %s seconds" % (str(time.time() - start)))
    if down_sample_rate < 1.:
        model.compute_metrics(X_val, y_val.ravel(), calibration_weight=down_sample_rate)
    else:
        model.compute_metrics(X_val, y_val.ravel())

    model.compute_features_distribution()



    if k < len(model.sorted_important_features):
        golden_features_tree = model.sorted_important_features[:k]
        print("Top %d strongly features %s" % (k, golden_features_tree))
        # df_num_corr = ensemble_train.corr()['click'][:-1]  # -1 because the latest row is SalePrice
        # golden_features_corr = df_num_corr[abs(df_num_corr) > 0.03].sort_values(ascending=False)
        # print(
        # "There is {} strongly correlated values with click:\n{}".format(len(golden_features_corr), golden_features_corr))

        golden_features = golden_features_tree
        X_train, y_train = ensemble_train[golden_features].values, ensemble_train[y_label].values

        model.clf.fit(X_train, y_train.ravel())
        if down_sample_rate < 1.:
            model.compute_metrics(X_val, y_val.ravel(), calibration_weight=down_sample_rate)
        else:
            model.compute_metrics(X_val, y_val.ravel())

        model.compute_features_distribution(golden_features)

    model.save()

