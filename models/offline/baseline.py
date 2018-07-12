#coding:utf8
import os
import gc
import argparse

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score, accuracy_score


from Conf import features_to_train, classifiers

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
args = parser.parse_args()

if __name__ == '__main__':
    
    path = '../features'
    USE_SAMPLE = args.sample
    # USER_FEATURE_TRAIN_FILE = 'user_feature_train'
    # USER_FEATURE_TRAIN_FILE = USER_FEATURE_TRAIN_FILE + '_sample.csv' if USE_SAMPLE else USER_FEATURE_TRAIN_FILE + '.csv'
    # user_item_train = pd.read_csv(os.path.join(path, USER_FEATURE_TRAIN_FILE), sep='\t')

    # USER_FEATURE_TEST_FILE = 'user_feature_test'
    # USER_FEATURE_TEST_FILE = USER_FEATURE_TEST_FILE + '_sample.csv' if USE_SAMPLE else USER_FEATURE_TEST_FILE + '.csv'
    # user_item_test = pd.read_csv(os.path.join(path, USER_FEATURE_TEST_FILE), sep='\t')

    ALL_FEATURE_TRAIN_FILE = 'ensemble_feature_train'
    ALL_FEATURE_TRAIN_FILE = ALL_FEATURE_TRAIN_FILE + '_sample.csv' if USE_SAMPLE else ALL_FEATURE_TRAIN_FILE + '.csv'
    ensemble_train = pd.read_csv(os.path.join(path, ALL_FEATURE_TRAIN_FILE), sep='\t')

    ALL_FEATURE_TEST_FILE = 'ensemble_feature_test'
    ALL_FEATURE_TEST_FILE = ALL_FEATURE_TEST_FILE + '_sample.csv' if USE_SAMPLE else ALL_FEATURE_TEST_FILE + '.csv'
    ensemble_test = pd.read_csv(os.path.join(path, ALL_FEATURE_TEST_FILE), sep='\t')

    time_min = min(ensemble_train['time'].min(), ensemble_test['time'].min())
    time_max = max(ensemble_train['time'].max(), ensemble_test['time'].max())
    duration_min = min(ensemble_train['duration_time'].min(), ensemble_test['duration_time'].min())
    duration_max = max(ensemble_train['duration_time'].max(), ensemble_test['duration_time'].max())

    print('time_min: %d, time_max: %d' % (time_min, time_max))
    print('duration_min: %d, duration_max: %d' % (duration_min, duration_max))
    time_df = ensemble_train['time']
    ensemble_train['time'] = (time_df-time_min)/(time_max-time_min)
    duration_df = ensemble_train['duration_time']
    ensemble_train['duration_time'] = (duration_df-duration_min)/(duration_max-duration_min)


    time_df = ensemble_test['time']
    ensemble_test['time'] = (time_df-time_min)/(time_max-time_min)
    duration_df = ensemble_test['duration_time']
    ensemble_test['duration_time'] = (duration_df-duration_min)/(duration_max-duration_min)

    print(ensemble_train.info())

    all_features = list(ensemble_train.columns.values)
    print("all features")
    print(all_features)

    print("train features")
    print(features_to_train)
    X = ensemble_train.as_matrix(features_to_train)
    y = ensemble_train['click'].values

    del ensemble_train
    gc.collect()

    print(X.shape)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    X_t = ensemble_test.as_matrix(features_to_train)
    X_t = preprocessing.scale(X_t)
    print(X_t.shape)

    for name, clf in classifiers.iteritems():
        clf.fit(X_train, y_train)
        print("{:31} 测试集acc/recall: {:15}/{:15}".format(name, 
            accuracy_score(y_test, clf.predict(X_test)), recall_score(y_test, clf.predict(X_test), average='macro')))

        y_sub = clf.predict_proba(X_t)[:,1]
        submission = pd.DataFrame()
        submission['user_id'] = ensemble_test['user_id']
        submission['photo_id'] = ensemble_test['photo_id']
        submission['click_probability'] = y_sub
        submission['click_probability'] = submission['click_probability'].apply(lambda x: float('%.6f' % x))
        submission.to_csv('Sub-'+ '-'.join(name.split(' ')) + '.txt', sep='\t', index=False, header=False)

    for name, clf in classifiers.iteritems():
        try: 
            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            print('{}特征权值分布为: '.format(name))
            for f in range(X_train.shape[1]):
                print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        except AttributeError:
            print('{} has no feture_importances_'.format(name))



    for name, clf in classifiers.iteritems():
        # y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        try:
            y_score = clf.decision_function(X_test)
        except AttributeError:
            print('{} has no decision_function, use predict func.'.format(name))
            y_score = clf.predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test, y_score, sample_weight=None)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        print('{} ROC curve (area = {})'.format(name, roc_auc))
