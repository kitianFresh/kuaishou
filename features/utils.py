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