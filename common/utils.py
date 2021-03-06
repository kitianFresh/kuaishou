#coding:utf8

import os
import numpy as np
import pandas as pd
import random
import time
import functools
import scipy.special as special
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class FeatureMerger(object):
    
    def __init__(self, col_feature_dir, col_features_to_merge, fmt='csv', data_type='train', pool_type='process', num_workers=8):
        self.fmt = fmt
        self.data_type = data_type
        self.pool_type = pool_type
        self.num_workers = num_workers
        self.col_feature_dir = col_feature_dir
        self.col_features_to_merge = col_features_to_merge
    
    def merge(self):   
        tasks_args = []
        for feature in self.col_features_to_merge:
            if self.data_type == "":
                file = feature  + '.' + self.fmt
            else:    
                file = feature  + '_' + self.data_type + '.' + self.fmt
            args = (feature, os.path.join(self.col_feature_dir, file), self.fmt)
            tasks_args.append(args)

        start_time_1 = time.clock()
        
        pool = ThreadPoolExecutor if self.pool_type == 'thread' else ProcessPoolExecutor
        with pool(max_workers=self.num_workers) as executor:
            dfs = (df for df in executor.map(feature_reader,  tasks_args))
        print("%s pool reading execution in %s seconds" % (self.pool_type, str(time.clock() - start_time_1)))
        
        start_time_1 = time.clock()
        print('Merging data')
        merger = functools.partial(pd.merge, how='inner', on=['user_id', 'photo_id'])
        data = functools.reduce(merger, dfs)
        print("Merging data in memory execution in %s seconds" % (str(time.clock() - start_time_1)))
        return data

# 无法使用多进程，不采样的数据集df太大了，多进程pickle不了, 还是只能使用共享内存的方案，这样得自己写了，暂不考虑
def reducer(dfs):
    n = len(dfs)
    mid = n/2+1
    if n <= 2:
        merger = functools.partial(pd.merge, how='inner', on=['user_id', 'photo_id'])
        data = functools.reduce(merger, dfs)
        return data
    with ProcessPoolExecutor(max_workers=2) as executor:
        left = executor.submit(reducer, dfs[:mid])
        right = executor.submit(reducer, dfs[mid:])
        data = pd.merge(left.result(), right.result(), how='inner', on=['user_id', 'photo_id'])
    return data
    
uint64_cols = ['user_id', 'photo_id', 'time']
uint32_cols = ['playing_sum', 'browse_time_diff', 'duration_sum', 'key_words_num']
uint16_cols = ['browse_num', 'exposure_num', 'click_num', 'duration_time', 'like_num', 'follow_num', 'clicked_num']
uint8_cols = ['cover_length', 'man_num', 'woman_num', 'face_num', 'time_cate', 'duration_time_cate']
bool_cols = ['have_face_cate', 'click']
float32_cols = ['clicked_ratio','non_face_click_favor', 'face_click_favor', 'man_favor', 'woman_avg_age', 'playing_freq', 'woman_age_favor', 'woman_yen_value_favor', 'human_scale', 'woman_favor', 'click_freq', 'woman_cv_favor', 'man_age_favor', 'man_yen_value_favor', 'follow_ratio', 'man_scale', 'browse_freq', 'man_avg_age', 'man_cv_favor', 'man_avg_attr', 'playing_ratio', 'woman_scale', 'click_ratio', 'human_avg_age', 'woman_avg_attr', 'like_ratio', 'cover_length_favor', 'human_avg_attr', 'avg_tfidf', 'hour_click_ratio']
float64_cols = []


feature_dtype_map = {}
for name in uint64_cols:
    feature_dtype_map.update({name: 'uint64'})
for name in uint32_cols:
    feature_dtype_map.update({name: 'uint32'})
for name in uint16_cols:
    feature_dtype_map.update({name: 'uint16'})
for name in uint8_cols:
    feature_dtype_map.update({name: 'uint8'})
for name in bool_cols:
    feature_dtype_map.update({name: 'bool'})
for name in float32_cols:
    feature_dtype_map.update({name: 'float32'})
for name in float64_cols:
    feature_dtype_map.update({name: 'float64'})

def feature_reader(args):
    global feature_dtype_map
    feature, path, fmt = args
    if fmt == 'csv':
        dtype = feature_dtype_map.get(feature)
        print(feature, dtype)
        df = pd.read_csv(path, sep='\t', dtype = {feature: dtype})
    elif fmt == 'pkl':
        df = pd.read_pickle(path)
    elif fmt == 'h5':
        df = pd.read_hdf(path, 'table', mode='r')
    
    return df
        
# def feature_reader(args):
#     feature, path, fmt = args
#     df = read_data(path, fmt)
#     return df        

def read_data(path, fmt):
    '''
    return DataFrame by given format data
    '''
    if fmt == 'csv':
        df = pd.read_csv(path, sep='\t')
    elif fmt == 'pkl':
        df = pd.read_pickle(path)
    elif fmt == 'h5':
        df = pd.read_hdf(path, 'table', mode='r')
    return df

def store_data(df, path, fmt, sep='\t', index=False):
    if fmt == 'csv':
        df.to_csv(path, sep=sep, index=index)
    elif fmt == 'pkl':
        df.to_pickle(path)
    elif fmt == 'h5':
        clib = 'blosc'
#         df.to_hdf(path, 'table', mode='w', complevel=9, complib=clib)
        df.to_hdf(path, 'table', mode='w')
        print('to hdf..')
    return path
        
        
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def normalize_min_max(df, features):
    all_features = set(list(df.columns.values))
    features = list(all_features & set(features))
    print('----------------------------------min_max norm------------------------------\n%s' % features)
    df[features] = df[features].apply(lambda x: (x-x.min())/(x.max()-x.min()))

def normalize_z_score(df, features):
    all_features = set(list(df.columns.values))
    features = list(all_features & set(features))
    print('----------------------------------z_score norm------------------------------\n%s' % features)
    df[features] = df[features].apply(lambda x: (x-x.mean())/x.std())


class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        print(sample)
        I = []
        C = []
        for clk_rt in sample:
            imp = imp_upperbound
            clk = imp * clk_rt
            I.append(imp)
            C.append(clk)
        return I, C

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0

        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i]+alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i]-clks[i]+beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(numerator_alpha/denominator), beta*(numerator_beta/denominator)

    
############################################################################################################################################################################# Feature Manual Discretization  #####################################################

#################
# time_features #
#################
import datetime
# # 凌晨 1-5 早上 5-8 上午 8-12 中午 12-15 下午 15-17 晚上 17-23 深夜 23-1
# period_1_5 = (datetime.time(1,0,0), datetime.time(5,0,0))
# period_5_8 = (datetime.time(5,0,0), datetime.time(8,0,0))
# period_8_12 = (datetime.time(8,0,0), datetime.time(12,0,0))
# period_12_15 = (datetime.time(12,0,0), datetime.time(15,0,0))
# period_15_17 = (datetime.time(15,0,0), datetime.time(17,0,0))
# period_17_23 = (datetime.time(17,0,0), datetime.time(23,0,0))
# period_0_1 = (datetime.time(0,0,0), datetime.time(1,0,0))
# period_23_0 = (datetime.time(23,0,0), datetime.time(0,0,0))

# def time_discretization(ts):
#     global period_1_5, period_5_8, period_8_12, period_12_15, period_15_17, period_17_23, period_0_1, period_23_0
#     dt = pd.to_datetime(ts-8*3600*1000, utc=True, unit='ms')
# #     print(type(dt))
#     t = dt.time()
#     if period_1_5[0] <= t <= period_1_5[1]:
#         return 0
#     elif period_5_8[0] < t <= period_5_8[1]:
#         return 1
#     elif period_8_12[0] < t <= period_8_12[1]:
#         return 2
#     elif period_12_15[0] < t <= period_12_15[1]:
#         return 3
#     elif period_15_17[0] < t <= period_15_17[1]:
#         return 4
#     elif period_17_23[0] < t <= period_17_23[1]:
#         return 5
#     elif period_0_1[0] <= t < period_0_1[1] or period_23_0[0] < t:
#         return 6

import datetime
periods_24 = []
for i in range(24):
    periods_24.append(datetime.time(i,0,0))

def time_discretization(ts):
    global periods_24
    dt = pd.to_datetime(ts-8*3600*1000, utc=True, unit='ms')
    t = dt.time()
    if periods_24[0] <= t < periods_24[1]:
        return 0
    elif periods_24[1] <= t < periods_24[2]:
        return 1
    elif periods_24[2] < t < periods_24[3]:
        return 2
    elif periods_24[3] < t < periods_24[4]:
        return 3
    elif periods_24[4] < t < periods_24[5]:
        return 4
    elif periods_24[5] < t < periods_24[6]:
        return 5
    elif periods_24[6] <= t < periods_24[7]:
        return 6
    elif periods_24[7] <= t < periods_24[8]:
        return 7
    elif periods_24[8] <= t < periods_24[9]:
        return 8
    elif periods_24[9] <= t < periods_24[10]:
        return 9
    elif periods_24[10] <= t < periods_24[11]:
        return 10
    elif periods_24[11] <= t < periods_24[12]:
        return 11
    elif periods_24[12] <= t < periods_24[13]:
        return 12
    elif periods_24[13] <= t < periods_24[14]:
        return 13
    elif periods_24[14] <= t < periods_24[15]:
        return 14
    elif periods_24[15] <= t < periods_24[16]:
        return 15
    elif periods_24[16] <= t < periods_24[17]:
        return 16
    elif periods_24[17] <= t < periods_24[18]:
        return 17
    elif periods_24[18] <= t < periods_24[19]:
        return 18
    elif periods_24[19] <= t < periods_24[20]:
        return 19
    elif periods_24[20] <= t < periods_24[21]:
        return 20
    elif periods_24[21] <= t < periods_24[22]:
        return 21
    elif periods_24[22] <= t < periods_24[23]:
        return 22
    else:
        return 23
    
    
def datetime_discretization(dt):
    global period_1_5, period_5_8, period_8_12, period_12_15, period_15_17, period_17_23, period_0_1, period_23_0
    t = dt.time()
    if period_1_5[0] <= t <= period_1_5[1]:
        return 0
    elif period_5_8[0] < t <= period_5_8[1]:
        return 1
    elif period_8_12[0] < t <= period_8_12[1]:
        return 2
    elif period_12_15[0] < t <= period_12_15[1]:
        return 3
    elif period_15_17[0] < t <= period_15_17[1]:
        return 4
    elif period_17_23[0] < t <= period_17_23[1]:
        return 5
    elif period_0_1[0] <= t < period_0_1[1] or period_23_0[0] < t:
        return 6

#################
# user_features #
#################
def browse_num_discretization(num):
    if 0 <= num and num <= 100:
        return 0
    elif 100 < num and num <= 200:
        return 1
    elif 200 < num and num <= 500:
        return 2
    elif 500 < num and num <= 1000:
        return 3
    else:
        return 4
def click_num_discretization(num):
    if 0 <= num and num <= 50:
        return 0
    elif 50 < num and num <= 100:
        return 1
    elif 100 < num and num <= 200:
        return 2
    elif 200 < num and num <= 500:
        return 3
    elif 500 < num and num <= 1000:
        return 4
    else:
        return 5
def like_num_discretization(num):
    if 0 <= num and num <= 10:
        return 0
    elif 10 < num and num <= 20:
        return 1
    elif 20 < num and num <= 30:
        return 3
    elif 30 < num and num <= 40:
        return 4
    else:
        return 5
def follow_num_discretization(num):
    if 0 <= num and num <= 5:
        return 0
    elif 5 < num and num <= 10:
        return 1
    elif 10 < num and num <= 15:
        return 3
    elif 15 < num and num <= 30:
        return 4
    else:
        return 5
def playing_sum_discretization(num):
    if 0 <= num and num <= 2000:
        return 0
    elif 2000 < num and num <= 5000:
        return 1
    elif 5000 < num and num <= 10000:
        return 2
    elif 10000 < num and num <= 20000:
        return 3
    else:
        return 4
def duration_sum_discretization(num):
    if 1000 <= num and num <= 10000:
        return 0
    elif 10000 < num and num <= 20000:
        return 1
    elif 20000 < num and num <= 30000:
        return 2
    elif 30000 < num and num <= 40000:
        return 3
    elif 40000 < num and num <= 60000:
        return 4
    elif 60000 < num and num <= 80000:
        return 5
    else:
        return 6
def click_ratio_discretization(num):
    if 0 <= num and num <= 0.1:
        return 0
    elif 0.1 < num and num <= 0.2:
        return 1
    elif 0.2 < num and num <= 0.4:
        return 2
    elif 0.4 < num and num <= 0.6:
        return 3
    elif 0.6 < num and num <= 0.8:
        return 4
    else:
        return 5
def like_ratio_discretization(num):
    if 0 <= num and num <= 0.02:
        return 0
    elif 0.02 < num and num <= 0.04:
        return 1
    elif 0.04 < num and num <= 0.06:
        return 2
    elif 0.06 < num and num <= 0.1:
        return 3
    else:
        return 4
    
def follow_ratio_discretization(num):
    if 0 <= num and num <= 0.02:
        return 0
    elif 0.02 < num and num <= 0.04:
        return 1
    elif 0.04 < num and num <= 0.06:
        return 2
    else:
        return 3
def playing_ratio_discretization(num):
    if 0 <= num and num <= 0.2:
        return 0
    elif 0.2 < num and num <= 0.4:
        return 1
    elif 0.4 < num and num <= 0.6:
        return 2
    else:
        return 3
def face_favor_discretization(num):
    if 0 <= num and num <= 1:
        return 0
    elif 1 < num and num <= 2:
        return 1
    else:
        return 2
def man_favor_discretization(num):
    if 0 <= num and num <= 1:
        return 0
    elif 1 < num and num <= 2:
        return 1
    else:
        return 2
def woman_favor_discretization(num):
    if 0 <= num and num <= 1:
        return 0
    elif 1 < num and num <= 2:
        return 1
    else:
        return 2
def man_cv_favor_discretization(num):
    if 0 <= num and num <= 0.02:
        return 0
    elif 0.02 < num and num <= 0.04:
        return 1
    elif 0.04 < num and num <= 0.06:
        return 2
    else:
        return 3
def woman_cv_favor_discretization(num):
    if 0 <= num and num <= 0.03:
        return 0
    elif 0.03 < num and num <= 0.06:
        return 1
    elif 0.06 < num and num <= 0.1:
        return 2
    else:
        return 3
def man_age_favor_discretization(num):
    if 0 <= num and num <= 4:
        return 0
    elif 4 < num and num <= 8:
        return 1
    elif 8 < num and num <= 12:
        return 2
    else:
        return 3

def woman_age_favor_discretization(num):
    if 0 <= num and num <= 4:
        return 0
    elif 4 < num and num <= 8:
        return 1
    elif 8 < num and num <= 12:
        return 2
    else:
        return 3

def man_yen_value_favor_discretization(num):
    if 0 <= num and num <= 10:
        return 0
    elif 10 < num and num <= 20:
        return 1
    else:
        return 2
    
def woman_yen_value_favor_discretization(num):
    if 0 <= num and num <= 10:
        return 0
    elif 10 < num and num <= 20:
        return 1
    elif 20 < num and num <= 40:
        return 2
    else:
        return 3
    
def non_face_click_favor_discretization(num):
    if 0 <= num <= 0.1:
        return 0
    elif 0.1 < num <= 0.2:
        return 1
    elif 0.2 < num <= 0.4:
        return 2
    elif 0.4 < num <= 0.6:
        return 3
    elif 0.6 < num <= 0.8:
        return 4
    else:
        return 5
    
def face_click_favor_discretization(num):
    if 0 <= num <= 0.1:
        return 0
    elif 0.1 < num <= 0.2:
        return 1
    elif 0.2 < num <= 0.4:
        return 2
    elif 0.4 < num <= 0.6:
        return 3
    elif 0.6 < num <= 0.8:
        return 4
    else:
        return 5
    
    
def playing_freq_discretization(num):
    if 0 <= num <= 0.05:
        return 0
    elif 0.05 < num <= 0.1:
        return 1
    elif 0.15 < num <= 0.2:
        return 2
    else:
        return 3

def browse_freq_discretization(num):
    if 0 <= num <= 0.02:
        return 0
    elif 0.02 < num <= 0.04:
        return 1
    elif 0.04 < num <= 0.06:
        return 2
    elif 0.06 < num <= 0.08:
        return 3
    else:
        return 4
    
def click_freq_discretization(num):
    if 0 <= num <= 0.003:
        return 0
    elif 0.003 < num <= 0.006:
        return 1
    elif 0.006 < num <= 0.01:
        return 2
    else:
        return 3

def cover_length_favor_discretization(num):
    if num == 0:
        return 0
    elif 0 < num <= 3:
        return 1
    elif 3 < num <= 10:
        return 2
    else:
        return 3
    
def browse_time_diff_discretization(num):
    if 0 <= num <= 500:
        return 0
    elif 500 < num <= 10000:
        return 1
    elif 10000 < num <= 20000:
        return 2
    elif 20000 < num <= 30000:
        return 3
    elif 30000 < num <= 40000:
        return 4
    elif 40000 < num <= 50000:
        return 5
    elif 50000 < num <= 60000:
        return 6
    elif 60000 < num <= 80000:
        return 7
    elif 80000 < num <= 100000:
        return 8
    else:
        return 9
    
    
##################
# photo_features #
##################

def exposure_num_discretization(exposure_num):
    if 0 <= exposure_num and exposure_num <= 10:
        return 0
    elif 10 < exposure_num and exposure_num <= 100:
        return 1
    elif 100 < exposure_num and exposure_num <= 1000:
        return 2
    else:
        return 3
    
def duration_time_discretization(t):
    if 0 < t <= 30:
        return 0
    elif 30 < t <= 60:
        return 1
    elif 60 < t < 300:
        return 2
    else:
        return 3

def face_num_discretization(face_num):
    if face_num == 0:
        return 0
    elif face_num == 1:
        return 1
    elif face_num == 2:
        return 2
    elif face_num == 3:
        return 3
    else:
        return 4

def man_num_discretization(man_num):
    if man_num == 0:
        return 0
    elif man_num == 1:
        return 1
    elif man_num == 2:
        return 2
    elif man_num == 3:
        return 3
    else:
        return 4
    
def woman_num_discretization(woman_num):
    if woman_num == 0:
        return 0
    elif woman_num == 1:
        return 1
    elif woman_num == 2:
        return 2
    elif woman_num == 3:
        return 3
    else:
        return 4

def human_avg_age_discretization(human_age):
    if human_age == 0:
        return 0
    elif 0 < human_age and human_age <= 12:
        return 1
    elif 12 < human_age and human_age <= 18:
        return 2
    elif 18 < human_age and human_age <= 24:
        return 3
    elif 24 < human_age and human_age <= 32:
        return 4
    elif human_age > 32:
        return 5

def woman_avg_age_discretization(woman_age):
    if woman_age == 0:
        return 0
    elif 0 < woman_age and woman_age <= 12:
        return 1
    elif 12 < woman_age and woman_age <= 18:
        return 2
    elif 18 < woman_age and woman_age <= 24:
        return 3
    elif 24 < woman_age and woman_age <= 28:
        return 4
    elif woman_age > 28:
        return 5

def man_avg_age_discretization(man_age):
    if man_age == 0:
        return 0
    elif 0 < man_age and man_age <= 12:
        return 1
    elif 12 < man_age and man_age <= 18:
        return 2
    elif 18 < man_age and man_age <= 24:
        return 3
    elif 24 < man_age and man_age <= 40:
        return 4
    elif woman_age > 40:
        return 5
    
def human_avg_attr_discretization(human_attr):
    if human_attr == 0:
        return 0
    elif 0 < human_attr and human_attr <= 50:
        return 1
    elif 50 < human_attr and human_attr <= 60:
        return 2
    elif 60 < human_attr and human_attr <= 70:
        return 3
    elif 70 < human_attr and human_attr <= 80:
        return 4
    elif human_attr > 80:
        return 5

def man_avg_attr_discretization(man_attr):
    if man_attr == 0:
        return 0
    elif 0 < man_attr and man_attr <= 50:
        return 1
    elif 50 < man_attr and man_attr <= 60:
        return 2
    elif 60 < man_attr and man_attr <= 70:
        return 3
    elif 70 < man_attr and man_attr <= 80:
        return 4
    elif man_attr > 80:
        return 5
    
def woman_avg_attr_discretization(woman_attr):
    if woman_attr == 0:
        return 0
    elif 0 < woman_attr and woman_attr <= 50:
        return 1
    elif 50 < woman_attr and woman_attr <= 60:
        return 2
    elif 60 < woman_attr and woman_attr <= 70:
        return 3
    elif 70 < woman_attr and woman_attr <= 80:
        return 4
    elif woman_attr > 80:
        return 5
    
def human_scale_discretization(human_scale):
    if human_scale == 0:
        return 0
    elif 0 < human_scale and human_scale <= 0.1:
        return 1
    elif 0.1 < human_scale and human_scale <= 0.2:
        return 2
    elif 0.2 < human_scale and human_scale <= 0.3:
        return 3
    elif 0.3 < human_scale and human_scale <= 0.4:
        return 4
    elif 0.4 < human_scale and human_scale <= 0.5:
        return 5
    elif 0.5 < human_scale and human_scale <= 0.6:
        return 6
    else:
        return 7

def man_scale_discretization(man_scale):
    if man_scale == 0:
        return 0
    elif 0 < man_scale and man_scale <= 0.1:
        return 1
    elif 0.1 < man_scale and man_scale <= 0.2:
        return 2
    elif 0.2 < man_scale and man_scale <= 0.3:
        return 3
    elif 0.3 < man_scale and man_scale <= 0.4:
        return 4
    elif 0.4 < man_scale and man_scale <= 0.5:
        return 5
    elif 0.5 < man_scale and man_scale <= 0.6:
        return 6
    else:
        return 7

def woman_scale_discretization(woman_scale):
    if woman_scale == 0:
        return 0
    elif 0 < woman_scale and woman_scale <= 0.1:
        return 1
    elif 0.1 < woman_scale and woman_scale <= 0.2:
        return 2
    elif 0.2 < woman_scale and woman_scale <= 0.3:
        return 3
    elif 0.3 < woman_scale and woman_scale <= 0.4:
        return 4
    elif 0.4 < woman_scale and woman_scale <= 0.5:
        return 5
    elif 0.5 < woman_scale and woman_scale <= 0.6:
        return 6
    else:
        return 7

def cover_length_discretization(num):
    if num == 0:
        return 0
    elif 0 < num <= 3:
        return 1
    elif 3 < num <= 10:
        return 2
    elif 10 < num <= 20:
        return 3
    else:
        return 4
    
def clicked_ratio_discretization(num):
    if 0 <= num <= 0.06:
        return 0
    elif 0.06 < num <= 0.12:
        return 1
    elif 0.12 < num <= 0.2:
        return 2
    elif 0.2 < num <= 0.3:
        return 3
    else:
        return 4