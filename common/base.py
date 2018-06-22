#coding:utf8

#from __future__ import unicode_literals
#from builtins import str as text
import functools
import os
import sys
import io
import json
import logging


import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import recall_score, roc_auc_score, precision_score, accuracy_score


from conf.modelconf import feature_dtype_map
from common.utils import FeatureMerger




class TransformerMixin(object):

    def __init__(self):
        pass

    def transform(self, func, *args, **kwargs):
        '''
        :param func: a transform function, this function use feature dataframe or table dataframe to do compute by default
        :param args:
        :param kwargs:
        :return: return a new Feature by transform a Feature or compute a Feature by table's features
        '''
        feat = func(args, kwargs)
        return feat

class Feature(TransformerMixin):

    _index_columns = ['user_id', 'photo_id']

    def __init__(self, feature_name, feature_dtype=None, feature_type='online', feature_dir='../data/features/columns', fmt='csv', feature_data=None):
        '''

        :param feature_name:
        :param feature_type: train or test or both, default for train and test.
        :param feature_dir:
        :param fmt:
        '''
        self.dir = feature_dir
        self.feature_type = feature_type
        self.feature_dtype = feature_dtype
        self.name = feature_name
        self.fmt = fmt
        self.df = self.load() if feature_data is None else feature_data

    def __read_data(self, path, fmt='csv'):
        '''
            return DataFrame by given format data
        '''
        if fmt == 'csv':
            if self.feature_dtype is None:
                dtype = feature_dtype_map.get(self.name)
            else:
                dtype = self.feature_dtype
            logging.info("(%s, %s)" % (self.name, dtype))
            df = pd.read_csv(path, sep='\t', dtype={self.name: dtype})
        elif fmt == 'pkl':
            df = pd.read_pickle(path)
        elif fmt == 'h5':
            df = pd.read_hdf(path, 'table', mode='r')
        else:
            raise IOError('fmt not found')
        return df

    def load(self):
        if self.df == None:
            path = os.path.join(self.dir, self.name+'_' + self.feature_type + '.' + self.fmt)
            self.df = self.__read_data(path, self.fmt)
            logging.info('Feature %s loading from %s' % (self.name, path))
        return self.df

    def feed(self, data):
        self.df = data

    def __store_data(self, path, fmt='csv', sep='\t', index=False):
        if fmt == 'csv':
            self.df.to_csv(path, sep=sep, index=index)
        elif fmt == 'pkl':
            self.df.to_pickle(path)
        elif fmt == 'h5':
            # clib = 'blosc'
            # df.to_hdf(path, 'table', mode='w', complevel=9, complib=clib)
            self.df.to_hdf(path, 'table', mode='w')
        return path

    def save(self):
        path = os.path.join(self.dir, self.name + '_' + self.feature_type + '.' + self.fmt)
        if self.df is not None:
            logging.info('Feature %s saving to %s' % (self.name, path))
            return self.__store_data(path, self.fmt)
        else:
            raise IOError('No data to store')



class Table(TransformerMixin):
    _merge_how = 'left'
    _index_columns = ['user_id', 'photo_id']

    _columns = []
    _table_type = 'train'

    def __init__(self, name, features=None,
                 table_dir='../data/features',
                 table_type=None,
                 table_fmt='csv',
                 col_feature_dir='../data/features/columns', pool_type='process', num_workers=8):
        '''
        :param name: table name, if `features` is None, it will try to read a table with the `name` in `table_dir`, and then combine it with features given.
        :param features: features used to composite this table, if not None, it will overwrite `_columns`.
                        if features element type is Feature in memory, it will composite a table with the name given.
                        if features element is str, it will read from disk and combine with the table in `table_dir`.

        :param table_dir: table dir to store to and read from
        :param table_type: online or offline(train or test)
        :param table_fmt:
        :param col_feature_dir: column features to store and read
        :param pool_type: pool type to use columns features to merge
        :param num_workers: worker nums to use columns features to merge
        '''

        self.name = name
        self.table_type = table_type if table_type else self._table_type
        self.table_dir = table_dir
        self.table_fmt = table_fmt
        self.features = {}
        if features and self.__check_feature_valid(features):
            for feat in features:
                self.features.update({feat.name: feat})
            self.df = self.__merge(how=self._merge_how, on=self._index_columns, features=features)
        else:
            # use a table and some column features.
            path = os.path.join(self.table_dir, name + '-' + self._table_type + '.' + self.table_fmt)
            if features:
                fm = FeatureMerger(col_feature_dir=col_feature_dir,
                               col_features_to_merge=features)
                feats_table = fm.merge()
            elif len(self._columns) > 0:
                fm = FeatureMerger(col_feature_dir=col_feature_dir,
                                   col_features_to_merge=self._columns)
                feats_table = fm.merge()
            else:
                feats_table = None

            if os.path.exists(path):
                one_table = self.__read_data(path, self.table_fmt)
            else:
                one_table = None
            if feats_table is not None and one_table is not None:
                self.df = self.__merge(how=self._merge_how, on=self._index_columns, features=[one_table, feats_table])
            elif feats_table is None and one_table is not None:
                self.df = one_table
            elif feats_table is not None and one_table is None:
                self.df = feats_table
            else:
                raise Exception('No table named %s or no features %s' % (self.name, features))

    def __read_data(self, path, fmt='csv'):
        '''
            return DataFrame by given format data
        '''
        if fmt == 'csv':
            df = pd.read_csv(path, sep='\t', dtype=feature_dtype_map)
        elif fmt == 'pkl':
            df = pd.read_pickle(path)
        elif fmt == 'h5':
            df = pd.read_hdf(path, 'table', mode='r')
        else:
            raise IOError('fmt not found')
        return df

    def add_feature(self, feature):
        if feature.name not in self.features:
            self.features.update({feature.name: feature})
            self.df = self.__merge(how=self._merge_how, on=self._index_columns, features=[self.df, feature.df])

    def add_features(self, features):
        if features and self.__check_feature_valid(features):
            for feat in features:
                self.features.update({feat.name: feat})
            df = self.__merge(how=self._merge_how, on=self._index_columns, features=features)
            self.df = self.__merge(how=self._merge_how, on=self._index_columns, features=[self.df, df])

    def __check_feature_valid(self, features):
        for feat in features:
            if not isinstance(feat, Feature):
                return False
        return True

    def __merge(self, how, on, features=None):
        merger = functools.partial(pd.merge, how=how, on=on)
        return functools.reduce(merger, [feat.df for feat in features if feat.df is not None])

    def get_feature(self, feature_name):
        return self.features.get(feature_name, None)

    def save_feature(self, feature_name):
        self.get_feature(feature_name).save()

    def save_all_in_one(self):
        # merger = functools.partial(pd.merge(how=self._merge_how, on=self._index_columns))
        # self.df = functools.reduce(merger, [feat.df for _, feat in self.features.items()])
        path = os.path.join(self.table_dir, self.name + '_' + self.table_type + '.csv')
        self.df.to_csv(path, fmt=self.table_fmt, sep='\t', index=False)

    def save_all_in_split(self):
        for _, feat in self.features.items():
            feat.save()


class ModelMixin(object):

    def cross_validation(self, func, *args, **kwargs):

        # KFold cross validation
        logging.info('Model %s cross validation......' % (self.model_name))
        scores = func(args, kwargs)
        logging.info('K Fold scores: %s' % scores)
        logging.info("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() ** 2))

    def compute_metrics(self, X_test, y_test):
        # predict 返回的是（n_sample,）预测标签，0，1，2，3....
        # predict_proba 返回的是 (n_sample, k) k分类的概率，第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1。
        # 因此这里的2分类问题 predict_proba(X_test)[:, 1] 就是返回点击了的概率
        # decision_function 与参数decision_function_shape取'ovr'、'ovo'有关，是点到超平面的距离。程序首先是计算出'ovo'结果，然后聚合成'ovr'结果
        self.precision = precision_score(y_test, self.clf.predict(X_test))
        self.recall = recall_score(y_test, self.clf.predict(X_test), average='macro')
        logging.info("{:31} 测试集precision/recall: {:15}/{:15}".format(self.model_name, self.precision, self.recall))
        self.accuracy = accuracy_score(y_test, self.clf.predict(X_test))
        logging.info("{:31} 测试集accuracy: {:15}".format(self.model_name, self.accuracy))

        if hasattr(self.clf, "decision_function"):
            y_score = self.clf.decision_function(X_test)
        else:
            logging.warning('{} has no decision_function, use predict func.'.format(self.model_name))
            y_score = self.clf.predict_proba(X_test)[:, 1]


        # Compute ROC curve and ROC area for each class
        self.roc_auc = roc_auc_score(y_test, y_score, sample_weight=None)
        # Plot ROC curve
        logging.info('{} ROC curve (area = {})'.format(self.model_name, self.roc_auc))

    def compute_features_distribution(self):
        '''

        :return: feature distribution list
        '''
        self.features_distribution = []
        self.sorted_important_features = []
        try:
            importances = self.clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            logging.info('{}特征权值分布为: '.format(self.model_name))
            for f in range(len(self.features_to_train)):
                logging.info("%d. feature %d [%s] (%f)" % (
                    f + 1, indices[f], self.features_to_train[indices[f]], importances[indices[f]]))

                self.features_distribution.append(
                    (f + 1, indices[f], self.features_to_train[indices[f]], importances[indices[f]]))

                self.sorted_important_features.append(self.features_to_train[indices[f]])
            logging.info(self.sorted_important_features)
        except AttributeError:
            logging.warning('{} has no feture_importances_'.format(self.model_name))


    def save(self):
        joblib.dump(self.clf, self.model_file)
        logging.info('Model %s saved in %s' % (self.model_name, self.model_file))
        model_metainfo = {
            'sub_file': self.sub_file,
            'model_file': self.model_file,
            'model_name': self.model_name,
            'version': self.version,
            'description': self.description,
            'features_distribution': self.features_distribution,
            'sorted_important_features': self.sorted_important_features,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'roc_auc': self.roc_auc,
        }

        # for python 2 and python 3 compatible, python 3 has a bug for this. https://stackoverflow.com/questions/11942364/typeerror-integer-is-not-json-serializable-when-serializing-json-in-python
        def default(o):
            if isinstance(o, np.int64): return int(o)
            raise TypeError
        # ensure_ascii=False 保证输出的不是 unicode 编码形式，而是真正的中文文本
        # from __future__ import unicode_literals
        with io.open(self.meta_info_file, mode='w', encoding='utf8') as outfile:
            metadata = json.dumps(model_metainfo, ensure_ascii=False, indent=4, default=default)
            # metadata is str in python2 which is actually bytearray, but in python3 all str is unicode. for compatibilty
            if sys.version_info < (3,):
                outfile.write(metadata.decode('utf8'))
            else:
                outfile.write(metadata)
            logging.info('Model %s meta info saved in %s' % (self.model_name, self.meta_info_file))

    def submit(self, ensemble_online, sparse_matrix=None):
        submission = pd.DataFrame()
        submission['user_id'] = ensemble_online['user_id']
        submission['photo_id'] = ensemble_online['photo_id']
        logging.info('Model %s make submission %s......' % (self.model_name, self.sub_file))
        X_t = sparse_matrix if sparse_matrix is not None else ensemble_online[self.features_to_train].values
        y_sub = []
        from sklearn.exceptions import NotFittedError
        try:
            y_sub = self.clf.predict_proba(X_t)[:, 1]
        except NotFittedError as e:
            logging.error(repr(e))
            exit(-1)
        submission['click_probability'] = y_sub
        submission['click_probability'] = submission['click_probability'].apply(lambda x: float('%.6f' % x))
        submission.to_csv(self.sub_file, sep='\t', index=False, header=False)


class Classifier(ModelMixin):

    def __init__(self, clf, dir, name, version, description, features_to_train, metric=None):
        '''
        :param clf:
        :param dir: model info dir
        :param name: model name
        :param version: model version
        :param description: model description, params or other info
        :param metric:
        '''

        self.model_name = name
        self.dir = os.path.join(dir, self.model_name)
        self.clf = clf
        self.version = version
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.sub_file = os.path.join(self.dir, 'sub-' + name + '-' + version + '.txt')
        self.model_file = os.path.join(self.dir, name + '-' + version + '.model')
        self.meta_info_file = os.path.join(self.dir, name + '-' + version + '.json')

        if os.path.exists(self.model_file) or \
            os.path.exists(self.meta_info_file) or \
            os.path.exists(self.sub_file):
            logging.warning(
                'There already has a model %s with the same version %s in %s' % (self.model_name, self.version, self.dir))
            sys.exit(-1)

        self.version = version
        self.description = description
        self.metric = metric
        self.features_to_train = features_to_train
        self.features_distribution = []
        self.sorted_important_features = []
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.roc_auc = None


