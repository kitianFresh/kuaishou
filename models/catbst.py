#coding:utf8
import os
import gc
import json
import io
import argparse
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import recall_score, accuracy_score
from catboost import CatBoostClassifier

from conf.modelconf import user_action_features, face_features, user_face_favor_features, id_features, time_features, photo_features, user_features, y_label, features_to_train

from common.utils import read_data, store_data, normalize_min_max, normalize_z_score, FeatureMerger

        
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-v', '--version', help='model version, there will be a version control and a json description file for this model', required=True)
parser.add_argument('-d', '--description', help='description for a model, a json description file attached to a model', required=True)

args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description
    
    model_name = 'CATBOOST'
    model_file = model_name + '-Sample' + '-' + version + '.model' if USE_SAMPLE else model_name + '-' + version + '.model'
    model_metainfo_file = model_name + '-Sample' + '-' + version + '.json' if USE_SAMPLE else model_name + '-' + version + '.json'
    sub_file = 'Sub-' + model_name + '-Sample' + '-' + version + '.txt' if USE_SAMPLE else 'Sub-' + model_name + '-' + version + '.txt'    
    if os.path.exists(model_file):
        print('There already has a model with the same version.')
        sys.exit(-1)
        
    
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'

    
#     ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
#     ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.' + fmt
#     ensemble_train = read_data(os.path.join(feature_store_path, ALL_FEATURE_TRAIN_FILE), fmt)

#     ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
#     ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.' + fmt
#     ensemble_test = read_data(os.path.join(feature_store_path, ALL_FEATURE_TEST_FILE), fmt)
    
    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'
    
    feature_to_use = user_features + photo_features + time_features
    
    fm_trainer = FeatureMerger(col_feature_store_path, feature_to_use+y_label, fmt=fmt, data_type='train', pool_type='process', num_workers=8)
    fm_tester = FeatureMerger(col_feature_store_path, feature_to_use, fmt=fmt, data_type='test', pool_type='process', num_workers=8)
    
    ensemble_train = fm_trainer.merge()
    print(ensemble_train.info())
    ensemble_test = fm_tester.merge()
    print(ensemble_test.info())

    all_features = list(ensemble_train.columns.values)
    print("all original features")
    print(all_features) 
    y = ensemble_train[y_label].values
    
    # less features to avoid overfit
    # features_to_train = ['exposure_num', 'click_ratio', 'cover_length_favor', 'woman_yen_value_favor', 'woman_cv_favor', 'cover_length', 'browse_num', 'man_age_favor', 'woman_age_favor', 'time', 'woman_scale', 'duration_time', 'woman_favor', 'playing_ratio', 'face_click_favor', 'click_num', 'man_cv_favor', 'man_scale', 'playing_sum', 'man_yen_value_favor', 'man_avg_age', 'playing_freq', 'woman_avg_attr', 'human_scale', 'browse_freq', 'non_face_click_favor', 'click_freq', 'woman_avg_age', 'human_avg_attr', 'duration_sum', 'man_favor', 'human_avg_age', 'follow_ratio', 'man_avg_attr']
    features_to_train = list(set(features_to_train) - set(['clicked_ratio']))
    submission = pd.DataFrame()
    submission['user_id'] = ensemble_test['user_id']
    submission['photo_id'] = ensemble_test['photo_id']
    

    print("train features")
    print(features_to_train)    

    ensemble_offline = ensemble_train[features_to_train]
    ensemble_online = ensemble_test[features_to_train]
    # 决策树模型不需要归一化，本身就是范围划分

    del ensemble_train
    del ensemble_test
    gc.collect()
    X = ensemble_offline.values
    print(X.shape)
    X_t = ensemble_online.values
    print(X_t.shape)
    del ensemble_offline
    del ensemble_online
    gc.collect()
    
    print('Training model %s......' % model_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = CatBoostClassifier(verbose=True)
    clf.fit(X_train, y_train.ravel())
    # KFold cross validation
    print('StratifiedKFold cross validation......')
    cv = StratifiedKFold(n_splits=3, random_state=0, shuffle=False)
    scores = cross_val_score(clf, X, y.ravel(), cv=cv, scoring='roc_auc')
    print('K Fold scores: %s' % scores)
    print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() ** 2))
    roc_auc = scores.mean()
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    recall = recall_score(y_test, clf.predict(X_test), average='macro')
    print("{:31} 测试集acc/recall: {:15}/{:15}".format(model_name, acc, recall))

    
    features_distribution = []
    important_features = []
    try: 
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print('{}特征权值分布为: '.format(model_name))
        for f in range(X_train.shape[1]):
            print("%d. feature %d [%s] (%f)" % (f + 1, indices[f], features_to_train[indices[f]], importances[indices[f]]))
            features_distribution.append((f + 1, indices[f], features_to_train[indices[f]], importances[indices[f]]))
            important_features.append(features_to_train[indices[f]])
        print(important_features)
    except AttributeError:
        print('{} has no feture_importances_'.format(model_name))


    try:
        y_score = clf.decision_function(X_test)[:,1]
    except AttributeError:
        print('{} has no decision_function, use predict func.'.format(model_name))
        y_score = clf.predict_proba(X_test)[:,1]

    # Compute ROC curve and ROC area for each class
    roc_auc = roc_auc_score(y_test, y_score, sample_weight=None)
    
    
    
    # Plot ROC curve
    print('{} ROC curve (area = {})'.format(model_name, roc_auc))
    
    print('Saving model %s to %s......' % (model_name, model_file))
    joblib.dump(clf, model_file)
    model_metainfo = {
        'sub_file': sub_file,
        'model_file': model_file,
        'model_name': model_name,
        'version': version,
        'description': desc,
        'features_to_train': features_to_train,
        'features_distribution': features_distribution,
        'important_features': important_features,
        'accuracy': acc,
        'recall': recall,
        'roc_auc': roc_auc,
    }
    #  ensure_ascii=False 保证输出的不是 unicode 编码形式，而是真正的中文文本
    with io.open(model_metainfo_file, 'w', encoding='utf8') as outfile:
        metadata = json.dumps(model_metainfo, outfile, ensure_ascii=False, indent=4)
        outfile.write(metadata.decode('utf8'))
    
    print('Make submission %s......' % sub_file)
    y_sub = clf.predict_proba(X_t)[:,1]
    submission['click_probability'] = y_sub
    submission['click_probability'] = submission['click_probability'].apply(lambda x: float('%.6f' % x))
    submission.to_csv(sub_file, sep='\t', index=False, header=False)
    
