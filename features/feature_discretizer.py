#coding:utf8

import os
import argparse
import sys
sys.path.append("..")
    
import pandas as pd
import numpy as np
from common import utils
from common.utils import *
from conf.modelconf import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sample', help='use sample data or full data', action="store_true")
parser.add_argument('-f', '--format', help='store pandas feature format, csv, pkl')
parser.add_argument('-t', '--data-type', help='data type, train or test, default all')
parser.add_argument('-c', '--column-discretization', action='append')

args = parser.parse_args()

if __name__ == '__main__':
    
    USE_SAMPLE = args.sample
    fmt = args.format if args.format else 'csv'
    dtypes = [args.data_type] if args.data_type else ['train', 'test']
    category_features = args.column_discretization if args.column_discretization else []
    
    feature_store_path = '../sample/features' if USE_SAMPLE else '../data/features'
    col_feature_store_path = '../sample/features/columns' if USE_SAMPLE else '../data/features/columns'
    
    
    for dtype in dtypes:
        for col in category_features:
            data_cate = pd.DataFrame()
            file = col + '_' + dtype + '.' + fmt
            data = read_data(os.path.join(col_feature_store_path, file), fmt)
            name = col+'_discretization'
            func = getattr(utils, name) if hasattr(utils, name) else None
            if func is not None and callable(func):
                print(func.__name__)
                data_cate[col+'_cate'] = data[col].apply(func)
            else:
                data_cate[col] = data[col]

            data_cate[col+'_cate'] = data_cate[col+'_cate'].astype('uint8')
            data_cate = pd.concat([data[id_features], data_cate], axis=1)

            print(data_cate.info())
            print(data_cate.head())

            CATE_FILE = col + '_cate_' + dtype + '.' + fmt
            store_data(data_cate, os.path.join(col_feature_store_path, CATE_FILE), fmt)

