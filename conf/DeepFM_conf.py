pre_path = '/Users/yuanguo/kuaishou/sample/offline/features/'

TEST_FILE = pre_path + 'ensemble_feature_test0.csv'
TRAIN_FILE = pre_path + 'ensemble_feature_train0.csv'

SUB_DIR = "."



NUM_SPLITS = 3
RANDOM_SEED = 2018

#CATEGORICAL_COLS = ['user_id', 'photo_id']
CATEGORICAL_COLS = []

NUMERIC_COLS = ['user_id', 'photo_id', 'click_ratio', 'browse_num',
                'click_num', 'like_num', 'follow_num', 'playing_sum',
                'duration_sum', 'like_ratio', 'follow_ratio',
                'playing_ratio', 'browse_time_diff', 'click_freq',
                'browse_freq', 'playing_freq', 'duration_favor',
                'face_favor', 'man_favor', 'woman_favor', 'man_cv_favor',
                'woman_cv_favor', 'man_age_favor', 'woman_age_favor',
                'man_yen_value_favor', 'woman_yen_value_favor', 'face_click_favor',
                'non_face_click_favor', 'cover_length_favor', 'exposure_num',
                'have_face_cate', 'have_text_cate', 'face_num', 'man_num',
                'woman_num', 'man_scale', 'woman_scale', 'human_scale',
                'man_avg_age', 'woman_avg_age', 'human_avg_age',
                'man_avg_attr', 'woman_avg_attr', 'human_avg_attr',
                'woman_num_ratio', 'man_num_ratio', 'cover_length',
                'avg_tfidf', 'key_words_num', 'text_class_label',
                'text_cluster_label', 'time', 'duration_time', 'click']

IGNORE_COLS = [ 'browse_time_diff', 'click_freq',
                'browse_freq', 'playing_freq',  'time']


import pandas as pd


class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict


    def scale(self, scale_fun, df_train, df_test=None):

        def scale_(df):

            for col in df:
                if col in self.feat_dict.ignore_cols:
                    continue
                if col in self.feat_dict.numeric_cols:
                    df[col] = (df[col]-df[col].min())/(df[col].max()-df[col].min())
            return df

        if not df_test is None:
            train_label = df_train.pop('click')
            test_label = df_test.pop('click')
            assert df_train.shape[1] == df_test.shape[1], \
                ('train columns %s != test columns %s ', df_train.shape[1], df_test.shape[1])
            train_line = df_train.shape[0]
            df = pd.concat([df_train, df_test])

            df = scale_(df)
            df_train = df.iloc[:train_line, :]
            df_test = df.iloc[train_line:, :]
            df_train['click'] = train_label
            df_test['click'] = test_label

            return df_train, df_test

        if df_test == None:
            return scale_(df_train)




    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        print(has_label)
        if has_label:
            y = dfi["click"].values.tolist()
            dfi.drop(["user_id", "photo_id", "click"], axis=1, inplace=True)
        else:
            ids = dfi[["user_id", "photo_id"]]
            dfi.drop(["user_id", "photo_id"], axis=1, inplace=True)
            dfi.drop(["click"], axis=1, inplace=True)

        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids

