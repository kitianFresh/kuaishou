#coding:utf8

import os
import numpy as np
import pandas as pd

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