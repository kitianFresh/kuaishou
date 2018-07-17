# coding:utf8

# must have one line in python3 between encoding and first import statement

import os
import sys
import time

sys.path.append("..")
import logging


import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier, Pool
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

from common.utils import dump_json_file


class TuningModelMixin(object):

    def config(self, name):
        logger = logging.getLogger(name)

        # 第一步，创建一个logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Log等级总开关

        # 第二步，创建一个handler，用于写入日志文件
        fh = logging.FileHandler(self.logging_file, mode='w')
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

        # 第三步，再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)  # 输出到console的log等级的开关
        # 第四步，定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 第五步，将logger添加到handler里面
        logger.addHandler(fh)
        logger.addHandler(ch)
        self.logging = logger


    def save_feature_importances(self, features, features_file):
        self.features_distribution = []
        self.features_to_train = features
        self.sorted_important_features = []
        try:
            importances = self.best_estimator.feature_importances_
            indices = np.argsort(importances)[::-1]
            logging.info('特征权值分布为: ')
            for f in range(len(self.features_to_train)):
                logging.info("%d. feature %d [%s] (%f)" % (
                    f + 1, indices[f], self.features_to_train[indices[f]], importances[indices[f]]))

                self.features_distribution.append(
                    (f + 1, indices[f], self.features_to_train[indices[f]], importances[indices[f]]))

                self.sorted_important_features.append(self.features_to_train[indices[f]])
            logging.info(self.sorted_important_features)
            data = {
                'features_distribution': self.features_distribution,
                'sorted_important_features': self.sorted_important_features,
            }
            dump_json_file(data, features_file)

        except AttributeError:
            logging.warning('no feture_importances_ attribute')


    def save(self,
             model_file,
             encode_model_file=None
             ):
        joblib.dump(self.best_estimator, model_file)
        if encode_model_file and self.grd_enc:
            joblib.dump(self.grd_enc, encode_model_file)

    def load(self,
             model_file,
             encode_model_file=None
             ):
        self.best_estimator = joblib.load(model_file)
        if encode_model_file:
            self.grd_enc = joblib.load(encode_model_file)
        else:
            self.grd_enc = None

    def save_best_params(self, best_params, params_file):
        dump_json_file(best_params, params_file)


    def predict_prob(self,
                     X):
        return self.best_estimator.predict_proba(X)


    def predict(self,
                X):
        return self.best_estimator.predict(X)


    def encode(self,
                X):
        return self.grd_enc.transform(X)


    def trainEvaluation(self):
        labels = self.y_test.ravel()
        predicts = self.predict(self.X_test)
        self.evaluation(labels=labels,
                        predicts=predicts
                        )


    def evaluation(self, labels, predicts):
        assert (labels.shape == predicts.shape)
        m = labels.shape
        c11 = 0.0
        c10 = 0.0
        c01 = 0.0
        c00 = 0.0
        count0 = 0.0
        count1 = 0.0
        for idx in range(0, m[0]):
            if labels[idx] == 0:
                count0 += 1
            else:
                count1 += 1
            if labels[idx] == 0 and predicts[idx] == 0:
                c00 += 1
            elif labels[idx] == 0 and predicts[idx] == 1:
                c01 += 1
            elif labels[idx] == 1 and predicts[idx] == 0:
                c10 += 1
            else:
                c11 += 1
        self.logging.info("score: {}".format((c00 + c11) / float(c00 + c01 + c11 + c10 + 0.001)))
        self.logging.info("precision: {}".format(c11 / float(c11 + c01 + 0.001)))
        self.logging.info("recall: {}".format(c11 / float(c11 + c10 + 0.001)))
        self.logging.info("positive: {}, negative: {}".format(c01 + c11, c10 + c00))
        return c11, c10, c01, c00



class gdbt_model(TuningModelMixin):

    def __init__(self,
                 X_train,
                 y_train,
                 X_test,
                 y_test,
                 random_state,
                 n_estimators,
                 learning_rate,
                 max_depth,
                 min_samples_split,
                 max_features,
                 min_samples_leaf,
                 subsample,
                 params_file,
                 logging_file,
                 n_jobs,
                 cv=2, iid=False, early_stopping_rounds=100):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.params_file = params_file
        self.logging_file = logging_file
        self.n_jobs = n_jobs
        self.cv = cv
        self.iid = iid
        self.early_stopping_rounds = early_stopping_rounds

    def process(self):
        self.config('gbdt')


    def train(self,
              date):
        start0 = time.time()

        fit_params = {
            'early_stopping_rounds': self.early_stopping_rounds,
            'eval_set': [[self.X_test, self.y_test]],
            'eval_metric': 'auc'
        }
        param_test = {'n_estimators': self.n_estimators, 'learning_rate': self.learning_rate}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(min_samples_split=500,
                                                 min_samples_leaf=100,
                                                 max_depth=5,
                                                 max_features='sqrt',
                                                 subsample=0.8,
                                                 random_state=self.random_state),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=self.iid, cv=self.cv)
        start = time.time()

        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search n_estimators and learning_rate result ......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))
        best_n_estimators = gsearch.best_params_['n_estimators']
        best_learning_rate = gsearch.best_params_['learning_rate']

        param_test = {'max_depth': self.max_depth, 'min_samples_split': self.min_samples_split}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                 n_estimators=best_n_estimators,
                                                 max_features='sqrt',
                                                 subsample=0.8,
                                                 random_state=self.random_state),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, fit_params=fit_params)
        self.logging.info("search max_depth and min_samples_split result ......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))
        best_max_depth = gsearch.best_params_['max_depth']
        best_min_samples_split = gsearch.best_params_['min_samples_split']

        param_test = {'min_samples_leaf': self.min_samples_leaf}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                 n_estimators=best_n_estimators,
                                                 subsample=0.8,
                                                 random_state=self.random_state,
                                                 max_depth=best_max_depth,
                                                 min_samples_split=best_min_samples_split,
                                                 ),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, fit_params=fit_params)
        self.logging.info("search min_samples_split result ......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))
        best_min_samples_leaf = gsearch.best_params_['min_samples_leaf']

        param_test = {'max_features': self.max_features}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                 n_estimators=best_n_estimators,
                                                 subsample=0.8,
                                                 random_state=self.random_state,
                                                 max_depth=best_max_depth,
                                                 min_samples_split=best_min_samples_split,
                                                 min_samples_leaf=best_min_samples_leaf),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, fit_params=fit_params)
        self.logging.info("search max_features result ......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))
        best_max_features = gsearch.best_params_['max_features']

        param_test = {'subsample': self.subsample}
        gsearch = GridSearchCV(
            estimator=GradientBoostingClassifier(learning_rate=best_learning_rate,
                                                 n_estimators=best_n_estimators,
                                                 max_features=best_max_features,
                                                 random_state=self.random_state,
                                                 max_depth=best_max_depth,
                                                 min_samples_split=best_min_samples_split),
            param_grid=param_test, scoring='roc_auc', n_jobs=self.n_jobs, iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, fit_params=fit_params)
        self.logging.info("search subsample result ......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))
        best_subsample = gsearch.best_params_['subsample']

        best_params = {"best_learning_rate":best_learning_rate,
                       "best_n_estimators":best_n_estimators,
                       "best_max_features":best_max_features,
                       "best_min_samples_leaf":best_min_samples_leaf,
                       "best_max_depth":best_max_depth,
                       "best_min_samples_split":best_min_samples_split,
                       "best_subsample":best_subsample,
                       "n_jobs":self.n_jobs
                       }

        self.save_best_params(best_params=best_params, params_file=self.params_file)
        self.logging.info({"best_learning_rate".format(best_learning_rate)})
        self.logging.info("best_n_estimators:{}".format(best_n_estimators))
        self.logging.info("best_max_features:{}".format(best_max_features))
        self.logging.info("best_min_samples_leaf:{}".format(best_min_samples_leaf))
        self.logging.info("best_max_depth:{}".format(best_max_depth))
        self.logging.info("best_min_samples_split:{}".format(best_min_samples_split))
        self.logging.info("best_subsample:{}".format(best_subsample))

        self.best_estimator = gsearch.best_estimator_

        temp_model = gsearch.best_estimator_
        test_predprob = temp_model.predict_proba(self.X_test)[:, 1]
        self.logging.info("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, test_predprob))
        self.logging.info("Total time: %s secs" % (time.time()-start0))

        # encode one-hot features
        grd_enc = OneHotEncoder()
        grd_enc.fit(temp_model.apply(self.X_train)[:, :, 0])
        self.grd_enc = grd_enc


class xgboost_model(TuningModelMixin):

    def __init__(self,
                 X_train,y_train,X_test,y_test,
                 random_state,
                 max_depth,
                 min_child_weight,
                 gamma,
                 subsample,
                 colsample_bytree,
                 reg_alpha,
                 n_estimators,
                 learning_rate,
                 params_file,
                 logging_file,
                 n_jobs, cv=2, iid=False, early_stopping_rounds=100):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv
        self.iid = iid
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.params_file = params_file
        self.logging_file = logging_file
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds

    def process(self):
        self.config('xgboost')


    def train(self,
              date):
        start0 = time.time()

        fit_params = {
            'early_stopping_rounds': self.early_stopping_rounds,
            'eval_set': [[self.X_test, self.y_test]],
            'eval_metric': 'auc'
        }

        param_test = {'learning_rate': self.learning_rate, "n_estimators": self.n_estimators}
        gsearch = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic',
                                                       nthread=self.n_jobs,
                                                       seed=self.random_state),
                               param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search reg_alpha......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_learning_rate = gsearch.best_params_['learning_rate']
        best_n_estimators = gsearch.best_params_['n_estimators']

        param_test = {'max_depth':self.max_depth,
                      'min_child_weight':self.min_child_weight}
        gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=best_learning_rate,
                                                        n_estimators=best_n_estimators,
                                                        objective='binary:logistic',
                                                        nthread=self.n_jobs,
                                                        seed=self.random_state),
                                param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search max_depth and min_child_weight......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_max_depth = gsearch.best_params_['max_depth']
        best_min_child_weight = gsearch.best_params_['min_child_weight']

        param_test = {'gamma': self.gamma}
        gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=best_learning_rate,
                                                        n_estimators=best_n_estimators,
                                                        max_depth=best_max_depth,
                                                        min_child_weight=best_min_child_weight,
                                                        objective='binary:logistic',
                                                        nthread=self.n_jobs,
                                                        seed=self.random_state),
                                param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search gamma......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_gamma = gsearch.best_params_['gamma']

        param_test = {'subsample': self.subsample,
                       'colsample_bytree': self.colsample_bytree}
        gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=best_learning_rate,
                                                       n_estimators=best_n_estimators,
                                                       max_depth=best_max_depth,
                                                       min_child_weight=best_min_child_weight,
                                                       gamma=best_gamma,
                                                       objective='binary:logistic',
                                                       nthread=self.n_jobs,
                                                       seed=self.random_state),
                                param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search subsample and colsample_bytree......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_subsample = gsearch.best_params_['subsample']
        best_colsample_bytree = gsearch.best_params_['colsample_bytree']

        param_test = {'reg_alpha': self.reg_alpha}
        gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=best_learning_rate,
                                                       n_estimators=best_n_estimators,
                                                       subsample=best_subsample,
                                                       colsample_bytree=best_colsample_bytree,
                                                       max_depth=best_max_depth,
                                                       min_child_weight=best_min_child_weight,
                                                       gamma=best_gamma,
                                                       objective='binary:logistic',
                                                       nthread=self.n_jobs,
                                                       seed=self.random_state),
                               param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search reg_alpha......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_reg_alpha = gsearch.best_params_['reg_alpha']



        best_params = {"best_max_depth": best_max_depth,
                       "best_min_child_weight": best_min_child_weight,
                       "best_subsample": best_subsample,
                       "best_colsample_bytree": best_colsample_bytree,
                       "best_reg_alpha": best_reg_alpha,
                       "best_learning_rate": best_learning_rate,
                       "best_n_estimators": best_n_estimators,
                       "n_jobs": self.n_jobs
                       }
        self.save_best_params(best_params=best_params, params_file=self.params_file)

        self.logging.info({"best_max_depth".format(best_max_depth)})
        self.logging.info("best_min_child_weight:{}".format(best_min_child_weight))
        self.logging.info("best_subsample:{}".format(best_subsample))
        self.logging.info("best_colsample_bytree:{}".format(best_colsample_bytree))
        self.logging.info("best_reg_alpha:{}".format(best_reg_alpha))
        self.logging.info("best_learning_rate:{}".format(best_learning_rate))
        self.logging.info("best_n_estimators:{}".format(best_n_estimators))
        self.logging.info("n_jobs:{}".format(self.n_jobs))

        self.best_estimator = gsearch.best_estimator_

        temp_model = gsearch.best_estimator_
        test_predprob = temp_model.predict_proba(self.X_test)[:, 1]
        self.logging.info("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, test_predprob))
        self.logging.info("Total time: %s secs" % (time.time()-start0))
        # encode one-hot features
        grd_enc = OneHotEncoder()
        grd_enc.fit(temp_model.apply(self.X_train))
        self.grd_enc = grd_enc


class lgbm_model(TuningModelMixin):

    def __init__(self,
                 X_train,y_train,X_test,y_test,
                 random_state,
                 max_depth,
                 min_child_weight,
                 gamma,
                 subsample,
                 colsample_bytree,
                 reg_alpha,
                 n_estimators,
                 learning_rate,
                 params_file,
                 logging_file,
                 n_jobs, cv=2, iid=False, early_stopping_rounds=100):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv
        self.iid = iid
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.params_file = params_file
        self.logging_file = logging_file
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds

    def process(self):
        self.config('lgbm')


    def train(self,
              date):
        start0 = time.time()

        fit_params = {
            'early_stopping_rounds': self.early_stopping_rounds,
            'eval_set': [[self.X_test, self.y_test]],
            'eval_metric': 'auc'
        }

        param_test = {'learning_rate': self.learning_rate, "n_estimators": self.n_estimators}
        gsearch = GridSearchCV(estimator=LGBMClassifier(nthread=self.n_jobs,
                                                        seed=self.random_state),
                               param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search reg_alpha......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_learning_rate = gsearch.best_params_['learning_rate']
        best_n_estimators = gsearch.best_params_['n_estimators']


        param_test = {'max_depth':self.max_depth,
                      'min_child_weight':self.min_child_weight}
        gsearch = GridSearchCV(estimator=LGBMClassifier(learning_rate=best_learning_rate,
                                                        n_estimators=best_n_estimators,
                                                        nthread=self.n_jobs,
                                                        seed=self.random_state),
                                param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search max_depth and min_child_weight......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_max_depth = gsearch.best_params_['max_depth']
        best_min_child_weight = gsearch.best_params_['min_child_weight']

        param_test = {'gamma': self.gamma}
        gsearch = GridSearchCV(estimator=LGBMClassifier(learning_rate=best_learning_rate,
                                                        n_estimators=best_n_estimators,
                                                        max_depth=best_max_depth,
                                                        min_child_weight=best_min_child_weight,
                                                        nthread=self.n_jobs,
                                                        seed=self.random_state),
                                param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search gamma......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_gamma = gsearch.best_params_['gamma']

        param_test = {'subsample': self.subsample,
                       'colsample_bytree': self.colsample_bytree}
        gsearch = GridSearchCV(estimator=LGBMClassifier(learning_rate=best_learning_rate,
                                                       n_estimators=best_n_estimators,
                                                       max_depth=best_max_depth,
                                                       min_child_weight=best_min_child_weight,
                                                       gamma=best_gamma,
                                                       nthread=self.n_jobs,
                                                       seed=self.random_state),
                                param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search subsample and colsample_bytree......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_subsample = gsearch.best_params_['subsample']
        best_colsample_bytree = gsearch.best_params_['colsample_bytree']

        param_test = {'reg_alpha': self.reg_alpha}
        gsearch = GridSearchCV(estimator=LGBMClassifier(learning_rate=best_learning_rate,
                                                       n_estimators=best_n_estimators,
                                                       subsample=best_subsample,
                                                       colsample_bytree=best_colsample_bytree,
                                                       max_depth=best_max_depth,
                                                       min_child_weight=best_min_child_weight,
                                                       gamma=best_gamma,
                                                       nthread=self.n_jobs,
                                                       seed=self.random_state),
                               param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv)
        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search reg_alpha......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_reg_alpha = gsearch.best_params_['reg_alpha']


        best_params = {"best_max_depth": best_max_depth,
                       "best_min_child_weight": best_min_child_weight,
                       "best_subsample": best_subsample,
                       "best_colsample_bytree": best_colsample_bytree,
                       "best_reg_alpha": best_reg_alpha,
                       "best_learning_rate": best_learning_rate,
                       "best_n_estimators": best_n_estimators,
                       "n_jobs": self.n_jobs
                       }
        self.save_best_params(best_params=best_params, params_file=self.params_file)

        self.logging.info({"best_max_depth".format(best_max_depth)})
        self.logging.info("best_min_child_weight:{}".format(best_min_child_weight))
        self.logging.info("best_subsample:{}".format(best_subsample))
        self.logging.info("best_colsample_bytree:{}".format(best_colsample_bytree))
        self.logging.info("best_reg_alpha:{}".format(best_reg_alpha))
        self.logging.info("best_learning_rate:{}".format(best_learning_rate))
        self.logging.info("best_n_estimators:{}".format(best_n_estimators))
        self.logging.info("n_jobs:{}".format(self.n_jobs))

        self.best_estimator = gsearch.best_estimator_

        temp_model = gsearch.best_estimator_
        test_predprob = temp_model.predict_proba(self.X_test)[:, 1]
        self.logging.info("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, test_predprob))
        self.logging.info("Total time: %s secs" % (time.time()-start0))
        # encode one-hot features
        grd_enc = OneHotEncoder()
        grd_enc.fit(temp_model.apply(self.X_train))
        self.grd_enc = grd_enc


class catboost_model(TuningModelMixin):

    def __init__(self,
                 X_train,y_train,X_test,y_test,
                 params_file,
                 logging_file,
                 random_state,
                 iterations,
                 learning_rate,
                 depth,
                 rsm,
                 n_jobs,
                 cv=2,
                 iid=False,
                 early_stopping_rounds=100,
                 task_type='GPU',
                 gpu_cat_features_storage='CpuPinnedMemory',
                 pinned_memory_size=1073741824 * 8,
                 used_ram_limit='200gb',
                 boosting_type="Plain",
                 simple_ctr='Borders:Prior=0.2',
                 max_ctr_complexity=4,
                 cat_feat_inds=[],
        ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cv = cv
        self.iid = iid
        self.early_stopping_rounds = early_stopping_rounds
        self.task_type = task_type
        self.gpu_cat_features_storage = gpu_cat_features_storage
        self.pinned_memory_size = pinned_memory_size
        self.used_ram_limit = used_ram_limit
        self.boosting_type = boosting_type
        self.simple_ctr = simple_ctr
        self.max_ctr_complexity = max_ctr_complexity
        self.cat_feat_inds = cat_feat_inds
        self.params_file= params_file
        self.logging_file = logging_file
        self.random_state = random_state
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.rsm = rsm
        self.n_jobs = n_jobs

    def process(self):
        self.config('catboost')


    def train(self,
            date):
        start0 = time.time()
        # train_pool = Pool(self.X_train, self.y_train, cat_features=self.cat_feat_inds)
        # validate_pool = Pool(self.X_test, self.y_test, cat_features=self.cat_feat_inds)
        fit_params = {
            'eval_set': (self.X_test, self.y_test),
            'cat_features': self.cat_feat_inds,
        }

        param_test = {'iterations': self.iterations, 'depth': self.depth, 'learning_rate': self.learning_rate}

        gsearch = GridSearchCV(
            estimator=CatBoostClassifier(loss_function='Logloss',
                                         eval_metric="AUC",
                                         od_type='Iter',
                                         od_wait=self.early_stopping_rounds,
                                         verbose=True,
                                         l2_leaf_reg=3,
                                         thread_count=self.n_jobs,
                                         simple_ctr=self.simple_ctr,
                                         task_type=self.task_type,
                                         gpu_cat_features_storage=self.gpu_cat_features_storage,
                                         used_ram_limit=self.used_ram_limit,
                                         pinned_memory_size=self.pinned_memory_size,
                                         boosting_type=self.boosting_type,
                                         max_ctr_complexity=self.max_ctr_complexity
                                         ),
            param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv
            )
        start = time.time()
        # gsearch.fit(self.X_train, self.y_train, fit_params=fit_params, model__cat_features=self.cat_feat_inds)
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search iterations, depth and learning_rate ......................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_iterations = gsearch.best_params_["iterations"]
        best_depth = gsearch.best_params_["depth"]
        best_learning_rate = gsearch.best_params_["learning_rate"]

        param_test = {'rsm': self.rsm}
        gsearch = GridSearchCV(
            estimator=CatBoostClassifier(loss_function='Logloss',
                                         eval_metric='AUC',
                                         od_type='Iter',
                                         od_wait=self.early_stopping_rounds,
                                         verbose=True,
                                         l2_leaf_reg=3,
                                         iterations=best_iterations,
                                         depth=best_depth,
                                         learning_rate=best_learning_rate,
                                         thread_count=self.n_jobs,
                                         simple_ctr=self.simple_ctr,
                                         task_type=self.task_type,
                                         gpu_cat_features_storage=self.gpu_cat_features_storage,
                                         used_ram_limit=self.used_ram_limit,
                                         pinned_memory_size=self.pinned_memory_size,
                                         boosting_type=self.boosting_type,
                                         max_ctr_complexity=self.max_ctr_complexity
                                         ),
            param_grid=param_test, scoring='roc_auc', iid=self.iid, cv=self.cv
        )

        start = time.time()
        gsearch.fit(self.X_train, self.y_train, **fit_params)
        self.logging.info("search rsm ...................................")
        self.logging.info(gsearch.grid_scores_)
        self.logging.info(gsearch.best_params_)
        self.logging.info("score:{}".format(gsearch.best_score_))
        self.logging.info("time spent %s secs" % (time.time() - start))

        best_rsm = gsearch.best_params_["rsm"]

        best_params = {"best_iterations": best_iterations,
                       "best_depth": best_depth,
                       "best_learning_rate": best_learning_rate,
                       "best_rsm":best_rsm,
                       "n_jobs": self.n_jobs,
                       "simple_ctr": self.simple_ctr,
                       "gpu_cat_features_storage": self.gpu_cat_features_storage,
                       "used_ram_limit": self.used_ram_limit,
                       "pinned_memory_size": self.pinned_memory_size,
                       "boosting_type": self.boosting_type,
                       "max_ctr_complexity": self.max_ctr_complexity,
                       "cat_feat_inds": self.cat_feat_inds,
                       }
        self.save_best_params(best_params=best_params, params_file=self.params_file)

        self.best_estimator = gsearch.best_estimator_

        temp_model = gsearch.best_estimator_
        test_predprob = temp_model.predict_proba(self.X_test)[:, 1]
        self.logging.info("AUC Score (Test): %f" % metrics.roc_auc_score(self.y_test, test_predprob))
        self.logging.info("Total time: %s secs" % (time.time()-start0))
        self.grd_enc = None