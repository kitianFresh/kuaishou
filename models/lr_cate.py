#coding:utf8
import os
import gc
import argparse
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score, accuracy_score
from scipy import sparse as ssp
from scipy.stats import spearmanr

from conf.modelconf import user_action_features, face_features, user_face_favor_features, id_features, time_features, photo_features, user_features, y_label, features_to_train

from common.utils import read_data, store_data, normalize_min_max, normalize_z_score

        
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')

args = parser.parse_args()

if __name__ == '__main__':
    
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    
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
    
    features_to_train = ['browse_num_cate', 'click_num_cate', 'like_num_cate', 'follow_num_cate', 'playing_sum_cate', 'duration_sum_cate', 'click_ratio_cate', 'like_ratio_cate', 'follow_ratio_cate', 'playing_ratio_cate', 'face_favor_cate', 'man_favor_cate', 'woman_favor_cate', 'man_cv_favor_cate', 'woman_cv_favor_cate', 'man_age_favor_cate', 'woman_age_favor_cate', 'man_yen_value_favor_cate', 'woman_yen_value_favor_cate', 'exposure_num_cate', 'have_face_cate', 'face_num_cate', 'man_num_cate', 'woman_num_cate', 'man_scale_cate', 'woman_scale_cate', 'human_scale_cate', 'man_avg_age_cate', 'woman_avg_age_cate', 'human_avg_age_cate', 'man_avg_attr_cate', 'woman_avg_attr_cate', 'human_avg_attr_cate', 'time_cate', 'duration_time_cate']
    
    
    submission = pd.DataFrame()
    submission['user_id'] = ensemble_test['user_id']
    submission['photo_id'] = ensemble_test['photo_id']
    

    print("train features")
    print(features_to_train)    

    
    ensemble_train = ensemble_train[features_to_train]
    ensemble_test = ensemble_test[features_to_train]
    num_train, num_test = ensemble_train.shape[0], ensemble_test.shape[0]
    ensemble_data = pd.concat([ensemble_train, ensemble_test])
    # pd.get_dummies must be str type 
    ensemble_data = ensemble_data.applymap(str)
    ensemble_data = pd.get_dummies(ensemble_data)
    
    print('---------------------------------------------------ont hot --------------------------------------------')
    print(ensemble_data.head())
    
    train = ensemble_data.iloc[:num_train,:]
    test = ensemble_data.iloc[num_train:,:]
    X = train.as_matrix()
    print(X.shape)
    X_t = test.as_matrix()
    print(X_t.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = LogisticRegression(C=1)
    name = "LogisticRegression"
    clf.fit(X_train, y_train)
    print("{:31} 测试集acc/recall: {:15}/{:15}".format(name, 
        accuracy_score(y_test, clf.predict(X_test)), recall_score(y_test, clf.predict(X_test), average='macro')))

    y_sub = clf.predict_proba(X_t)[:,1]
    submission['click_probability'] = y_sub
    submission['click_probability'] = submission['click_probability'].apply(lambda x: float('%.6f' % x))
    dst = 'Sub-' + name + '-Sample.txt' if USE_SAMPLE else 'Sub-' + name + '.txt'
    submission.to_csv(dst, sep='\t', index=False, header=False)

    try: 
        important_features = []
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print('{}特征权值分布为: '.format(name))
        for f in range(X_train.shape[1]):
            print("%d. feature %d [%s] (%f)" % (f + 1, indices[f], features_to_train[indices[f]], importances[indices[f]]))
            important_features.append(features_to_train[indices[f]])
    except AttributeError:
        print('{} has no feture_importances_'.format(name))

    print(important_features)

    try:
        y_score = clf.decision_function(X_test)[:,1]
    except AttributeError:
        print('{} has no decision_function, use predict func.'.format(name))
        y_score = clf.predict_proba(X_test)[:,1]

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_score, sample_weight=None)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    print('{} ROC curve (area = {})'.format(name, roc_auc))
