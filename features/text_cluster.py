#coding=utf-8

"""
@author: coffee
@time: 18-6-25 上午9:24
"""

import sys
sys.path.append('..')
from common import cluster



def train_cluster_model(cluster_model_path,data,cluster_nums):
    print('start training {} class kmeans'.format(cluster_nums))
    clustered = cluster.train(data,cluster_nums,cluster_model_path)
    print('training finished')
    return clustered


def cluster_model_predict(model_path,data):
    return cluster.predict(data,model_path)
