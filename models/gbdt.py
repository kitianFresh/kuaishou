#coding:utf8
import os
import gc
import argparse
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


from conf.modelconf import user_action_features, face_features, user_face_favor_features, id_features, time_features, photo_features, user_features, y_label, features_to_train


def read_data(path, fmt):
    '''
    return DataFrame by given format data
    '''
    if fmt == 'csv':
        df = pd.read_csv(path, sep='\t')
    elif fmt == 'pkl':
        df = pd.read_pickle(path)
    return df

def store_data(df, path, fmt, sep='\t', index=False):
    if fmt == 'csv':
        df.to_csv(path, sep=sep, index=index)
    elif fmt == 'pkl':
        df.to_pickle(path)
        
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')

args = parser.parse_args()

if __name__ == '__main__':
    
    path = '../features'
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    
    ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
    ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.' + fmt
    ensemble_train = read_data(os.path.join(path, ALL_FEATURE_TRAIN_FILE), fmt)

    ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
    ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample' + '.' + fmt if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.' + fmt
    ensemble_test = read_data(os.path.join(path, ALL_FEATURE_TEST_FILE), fmt)

    print(ensemble_train.info())
    print(ensemble_test.info())

    all_features = list(ensemble_train.columns.values)
    print("all original features")
    print(all_features) 
    y = ensemble_train[y_label].values

    submission = pd.DataFrame()
    submission['user_id'] = ensemble_test['user_id']
    submission['photo_id'] = ensemble_test['photo_id']
#     features_to_train = ['click_ratio', 'exposure_num', 'click_num', 'like_ratio',
#  'playing_ratio', 'playing_sum', 'follow_ratio', 'woman_favor', 'woman_age_favor', 'woman_scale',
#  'woman_cv_favor', 'human_scale', 'browse_num', 'duration_sum']
    print("train features")
    print(features_to_train)    

    ensemble_train = ensemble_train[features_to_train]
    ensemble_test = ensemble_test[features_to_train]
    num_train, num_test = ensemble_train.shape[0], ensemble_test.shape[0]
    ensemble_data = pd.concat([ensemble_train, ensemble_test])
    def normalize(df, features):
        df[features] = df[features].apply(lambda x: (x-x.min())/(x.max()-x.min()))
    
    norm_features = ['exposure_num', 'click_num', 'playing_sum', 'woman_favor', 'woman_age_favor', 'woman_scale', 'woman_cv_favor', 'human_scale', 'browse_num', 'duration_sum']
    
    normalize(ensemble_data, norm_features)
    train = ensemble_data.iloc[:num_train,:]
    test = ensemble_data.iloc[num_train:,:]
    X = train.as_matrix()
    print(X.shape)
    X_t = test.as_matrix()
    print(X_t.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_leaf=9)
    name = "GBDT"
    clf.fit(X_train, y_train)
    print("{:31} 测试集acc/recall: {:15}/{:15}".format(name, 
        accuracy_score(y_test, clf.predict(X_test)), recall_score(y_test, clf.predict(X_test), average='macro')))

    y_sub = clf.predict_proba(X_t)[:,1]
    submission['click_probability'] = y_sub
    submission['click_probability'] = submission['click_probability'].apply(lambda x: float('%.6f' % x))
    dst = 'Sub-' + name + '-Sample.txt' if USE_SAMPLE else 'Sub-' + name + '.txt'
    submission.to_csv(dst, sep='\t', index=False, header=False)
    
    try: 
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        print('{}特征权值分布为: '.format(name))
        important_features = []
        for f in range(X_train.shape[1]):
            print("%d. feature %d [%s] (%f)" % (f + 1, indices[f], features_to_train[indices[f]], importances[indices[f]]))
            important_features.append(features_to_train[indices[f]])
    except AttributeError:
        print('{} has no feture_importances_'.format(name))
    print(important_features)


    # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    try:
        y_score = clf.decision_function(X_test)[:,1]
    except AttributeError:
        print('{} has no decision_function, use predict func.'.format(name))
        y_score = clf.predict_proba(X_test)[:,1]

    # Compute ROC curve and ROC area for each class
    roc_auc = roc_auc_score(y_test, y_score, sample_weight=None)

    # Plot ROC curve
    print('{} ROC curve (area = {})'.format(name, roc_auc))
