#coding:utf8
import os
import gc
import io
import json
import argparse
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score
from scipy import sparse as ssp
from scipy.stats import spearmanr

from conf.modelconf import user_action_features, face_features, user_face_favor_features, id_features, time_features, photo_features, user_features, y_label, features_to_train

from common.utils import read_data, store_data, normalize_min_max, normalize_z_score

        
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-v', '--version', help='model version, there will be a version control and a json description file for this model', required=True)
parser.add_argument('-d', '--description', help='description for a model, a json description file attached to a model', required=True)

args = parser.parse_args()


if __name__ == '__main__':
    
    path = '../features'
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    version = args.version
    desc = args.description
    
    model_name = 'LR-Cate'
    model_file = model_name + '-Sample' + '-' + version + '.model' if USE_SAMPLE else model_name + '-' + version + '.model'
    model_metainfo_file = model_name + '-Sample' + '-' + version + '.json' if USE_SAMPLE else model_name + '-' + version + '.json'
    sub_file = 'Sub-' + model_name + '-Sample' + '-' + version + '.txt' if USE_SAMPLE else 'Sub-' + model_name + '-' + version + '.txt'
    
    if os.path.exists(model_file):
        print('There already has a model with the same version.')
        sys.exit(-1)
        
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'

    
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
    
    features_to_train = list(set(all_features) - set(['user_id', 'photo_id', 'click']))
    
#     features_to_train = [name + '_cate' for name in features_to_train]
    submission = pd.DataFrame()
    submission['user_id'] = ensemble_test['user_id']
    submission['photo_id'] = ensemble_test['photo_id']
    

    print("train features")
    print(features_to_train)    

    
    ensemble_train = ensemble_train[features_to_train]
    ensemble_test = ensemble_test[features_to_train]
    num_train, num_test = ensemble_train.shape[0], ensemble_test.shape[0]
    ensemble_data = pd.concat([ensemble_train, ensemble_test])
    
    # 1. INSTANTIATE
    enc = preprocessing.OneHotEncoder()
    # 2. FIT
    enc.fit(ensemble_data)
    # 3. Transform
    ensemble_data = enc.transform(ensemble_data)

    # pd.get_dummies must be str type not a sparse matrix by pandas get dummies
#     ensemble_data = ensemble_data.applymap(str)
#     ensemble_data = pd.get_dummies(ensemble_data)
    
    train = ensemble_data[:num_train,:]
    test = ensemble_data[num_train:,:]
    X = train
    print(X.shape)
    X_t = test
    print(X_t.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = LogisticRegression(C=1)
    
    clf.fit(X_train, y_train.ravel())
    acc = accuracy_score(y_test, clf.predict(X_test))
    recall = recall_score(y_test, clf.predict(X_test), average='macro')
    print("{:31} 测试集acc/recall: {:15}/{:15}".format(model_name, acc, recall))

    y_sub = clf.predict_proba(X_t)[:,1]
    submission['click_probability'] = y_sub
    submission['click_probability'] = submission['click_probability'].apply(lambda x: float('%.6f' % x))
    submission.to_csv(sub_file, sep='\t', index=False, header=False)
    
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
        y_score = clf.decision_function(X_test)
    except AttributeError:
        print('{} has no decision_function, use predict func.'.format(model_name))
        y_score = clf.predict_proba(X_test)[:,1]

    # Compute ROC curve and ROC area for each class
    roc_auc = roc_auc_score(y_test, y_score.ravel(), sample_weight=None)
    # Plot ROC curve
    print('{} ROC curve (area = {})'.format(model_name, roc_auc))
    
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
    with io.open(model_metainfo_file, 'w', encoding='utf8') as outfile:
        metadata = json.dumps(model_metainfo, outfile, ensure_ascii=False, indent=4)
        outfile.write(metadata.decode('utf8'))
