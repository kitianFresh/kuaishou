import argparse
import datetime
import time
import os
import sys
sys.path.append('../../')
from multiprocessing import cpu_count

from models.gbm_auto_tuning import gdbt_model, lgbm_model, xgboost_model, catboost_model
from conf.modelconf import *
from common.utils import read_data, store_data, FeatureMerger

def parse_args():
    parser = argparse.ArgumentParser(description="GradientBoostingMachine Auto Tuning")
    parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
    parser.add_argument('-g', '--gpu-mode', help='use gpu mode or not, for lgbm, catboost', action="store_true")
    parser.add_argument('--all', help='use one ensemble table all, or merge by columns', action='store_true')
    parser.add_argument("--n_estimators", help="number of base weak estimators default=[100, 300, 500, 1000]", default=[100, 300, 500, 1000], type=list)
    parser.add_argument("--max_depth", help="max tree depth, default=[4, 5, 6, 7, 8, 9]", default=[4, 5, 6, 7, 8, 9], type=list)
    parser.add_argument("--min_samples_split", help="minimum samples to split, default=[30, 50, 70, 90]", default=[30, 50, 70, 90], type=list)
    parser.add_argument("--min_samples_leaf", help="minimum samples of a leaf, default=[20, 40, 60, 80, 100]", default=[20, 40, 60, 80, 100], type=list)
    parser.add_argument("--max_features", help="maximum features of each base estimator, default=['sqrt', 'log2', 'None']", default=['sqrt', 'log2', 'None'], type=list)
    parser.add_argument("--subsample", help="ramdom select subsample rate, default=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]", default=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], type=list)
    parser.add_argument("--random_state", help="ramdom seed, default=777", default=777, type=int)
    parser.add_argument("--n_jobs", help="job nums, default=cpu_count", default=cpu_count(), type=int)
    parser.add_argument('-l', '--descreate-max-num', help='catboost model category feature descreate_max_num, max=48',
                        default=30)
    parser.add_argument('-m', '--model', help='model to tuning, support lgbm, xgboost, catboost', action='append')

    return parser.parse_args()


def train(X_train, y_train, X_val, y_val,
          date,
          random_state,
          n_estimators,
          max_depth,
          min_samples_split,
          min_samples_leaf,
          max_features,
          subsample,
          cat_feature_inds,
          task_type,
          n_jobs,
          log_dir,
          params_dir,
          tuning_model_dir,
          tuning_encode_model_dir,
          models,
          ):

    print(models)
    for model in models:
        if model == 'gbdt':
            model1 = gdbt_model(X_train, y_train, X_val, y_val,
                               random_state=random_state,
                               learning_rate=[0.005, 0.01, 0.05, 0.1],
                               n_estimators=[100, 300, 500, 800, 1200, 1500, 2000],
                               max_depth=[4, 5, 6, 7, 8, 9],
                               min_samples_split=[20, 40, 60, 80],
                               min_samples_leaf=[40],
                               max_features=max_features,
                               subsample=[0.6, 0.7, 0.8, 0.9],
                               params_file=os.path.join(params_dir, "gdbt_params{}.json".format(date)),
                               logging_file=os.path.join(log_dir, "gbdtTuning{}.log".format(date)),
                               n_jobs=n_jobs)
            model1.process()
            model1.train(date=date)
            model1.save_feature_importances(features_to_train, os.path.join(params_dir, "gbdt_features{}.json".format(date)))
            model1.save(model_file=os.path.join(tuning_model_dir, "gbdt{}.model".format(date)),
                       encode_model_file=os.path.join(tuning_encode_model_dir, "gdbt_encode{}.model".format(date)))
            model1.trainEvaluation()
        elif model == 'lgbm':
            model2 = lgbm_model(X_train, y_train, X_val, y_val,
                                   random_state=random_state,
                                   max_depth=[4, 5, 6, 7, 8, 9],
                                   min_child_weight=[2, 4, 6, 8],
                                   gamma = [0.0, 0.1, 0.2, 0.3, 0.4],
                                   subsample=[0.6, 0.7, 0.8, 0.9],
                                   colsample_bytree=[0.6, 0.7, 0.8, 0.9],
                                   reg_alpha=[1e-5, 1e-2, 0.1],
                                   n_estimators=[100, 300, 500, 800, 1200, 1500, 2000],
                                   learning_rate=[0.001, 0.005, 0.01, 0.03, 0.05, 0.1],
                                   params_file=os.path.join(params_dir, "lgbm_params{}.json".format(date)),
                                   logging_file=os.path.join(log_dir, "lgbmTuning{}.log".format(date)),
                                   n_jobs=n_jobs
                                  )
            model2.process()
            model2.train(date=date)
            model2.save_feature_importances(features_to_train, os.path.join(params_dir, "lgbm_features{}.json".format(date)))
            model2.save(model_file=os.path.join(tuning_model_dir, "lgbm{}.model".format(date)),
                       encode_model_file=os.path.join(tuning_encode_model_dir, "lgbm{}.model".format(date)))
            model2.trainEvaluation()
        elif model == 'catboost':
            model3 = catboost_model(
                X_train, y_train, X_val, y_val,
                params_file=os.path.join(params_dir, "catboost_params{}.json".format(date)),
                logging_file=os.path.join(log_dir, "catboostTuning{}.log".format(date)),
                random_state=random_state,
                iterations=[100, 300, 500, 800, 1200, 1500, 2000],
                learning_rate=[0.001, 0.005, 0.01, 0.03, 0.05, 0.1],
                depth=[4, 5, 6, 7, 8, 9],
                rsm = [0.8],
                n_jobs=n_jobs,
                task_type=task_type,
            )
            model3.process()
            model3.train(date=date)
            model3.save_feature_importances(features_to_train, os.path.join(params_dir, "catboost_features{}.json".format(date)))
            model3.save(os.path.join(tuning_model_dir, "catboost{}.model".format(date)))
        elif model == 'xgboost':
            model4 = xgboost_model(X_train, y_train, X_val, y_val,
                                   random_state=random_state,
                                   max_depth=[4, 5, 6, 7, 8, 9],
                                   min_child_weight=[2, 4, 6, 8],
                                   gamma = [0.0, 0.1, 0.2, 0.3, 0.4],
                                   subsample=[0.6, 0.7, 0.8, 0.9],
                                   colsample_bytree=[0.6, 0.7, 0.8, 0.9],
                                   reg_alpha=[1e-5, 1e-2, 0.1],
                                   n_estimators=[100, 300, 500, 800, 1200, 1500, 2000],
                                   learning_rate=[0.001, 0.005, 0.01, 0.03, 0.05, 0.1],
                                   params_file=os.path.join(params_dir, "xgboost_params{}.json".format(date)),
                                   logging_file=os.path.join(log_dir, "xgboostTuning{}.log".format(date)),
                                   n_jobs=n_jobs
                                  )
            model4.process()
            model4.train(date=date)
            model4.save_feature_importances(features_to_train, os.path.join(params_dir, "xgboost_features{}.json".format(date)))
            model4.save(model_file=os.path.join(tuning_model_dir, "xgboost{}.model".format(date)),
                        encode_model_file=os.path.join(tuning_encode_model_dir, "xgboost_encode{}.model".format(date)))
            model4.trainEvaluation()
        else:
            print('Not supported model %s' % model)
            exit(0)


def main(args):

    n_estimators = args.n_estimators
    max_depth = args.max_depth
    min_samples_split = args.min_samples_split
    min_samples_leaf = args.min_samples_leaf
    max_features = args.max_features
    subsample = args.subsample
    random_state = args.random_state
    n_jobs = args.n_jobs
    models = args.model
    gpu_mode = args.gpu_mode
    fmt = args.format if args.format else 'csv'
    all_one = args.all
    kfold = 0

    model_store_dir = './sample/' if USE_SAMPLE else './data'

    feature_store_dir = os.path.join(offline_data_dir, 'features')
    col_feature_store_dir = os.path.join(feature_store_dir, 'columns')
    log_dir = os.path.join(model_store_dir, 'log')
    params_dir = os.path.join(model_store_dir, 'param')
    tuning_model_dir = os.path.join(model_store_dir, 'model')
    tuning_encode_model_dir = os.path.join(model_store_dir, 'encode_model')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(params_dir):
        os.mkdir(params_dir)
    if not os.path.exists(tuning_model_dir):
        os.mkdir(tuning_model_dir)
    if not os.path.exists(tuning_encode_model_dir):
        os.mkdir(tuning_encode_model_dir)

    start = time.time()
    if all_one:
        ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train' + str(kfold) + '.' + fmt
        ensemble_train = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TRAIN_FILE), fmt)

        ALL_FEATURE_TEST_FILE = 'ensemble_feature_test' + str(kfold) + '.' + fmt
        ensemble_test = read_data(os.path.join(feature_store_dir, ALL_FEATURE_TEST_FILE), fmt)
    else:
        feature_to_use = id_features + user_features + photo_features + time_features
        fm_trainer = FeatureMerger(col_feature_store_dir, feature_to_use+y_label, fmt=fmt, data_type='train', pool_type='process', num_workers=num_workers)
        fm_tester = FeatureMerger(col_feature_store_dir, feature_to_use+y_label, fmt=fmt, data_type='test', pool_type='process', num_workers=num_workers)
        ensemble_train = fm_trainer.concat()
        ensemble_test = fm_tester.concat()

    end = time.time()
    print('data read in %s seconds' % str(end - start))

    print(ensemble_train.info())
    print(ensemble_test.info())

    X_train, y_train = ensemble_train[features_to_train].values, ensemble_train[y_label].values

    X_val, y_val = ensemble_test[features_to_train].values, ensemble_test[y_label].values

    print(y_train.mean(), y_train.std())
    print(y_val.mean(), y_val.std())

    cat_feature_inds = []
    descreate_max_num = 30
    descreate_max_num = 48 if descreate_max_num > 48 else descreate_max_num
    # for i, c in enumerate(ensemble_train[features_to_train].columns):
    #     num_uniques = ensemble_train[features_to_train][c].nunique()
    #     if num_uniques < descreate_max_num:
    #         print(c, num_uniques, descreate_max_num)
    #         cat_feature_inds.append(i)
    cat_feature_inds = [27, 28, 29, 30, 31, 46, 47]

    train(X_train, y_train.ravel(), X_val, y_val.ravel(),
              date=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
              random_state=random_state,
              n_estimators=n_estimators,
              max_depth=max_depth,
              min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf,
              max_features=max_features,
              subsample=subsample,
              n_jobs=n_jobs,
              log_dir=log_dir,
              params_dir=params_dir,
              tuning_model_dir=tuning_model_dir,
              tuning_encode_model_dir=tuning_encode_model_dir,
              cat_feature_inds=cat_feature_inds,
              task_type='GPU' if gpu_mode else 'CPU',
              models=models
              )

if __name__ == "__main__":
    args = parse_args()
    main(args)