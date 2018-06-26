#coding=utf-8

"""
@author: coffee
@time: 18-6-25 上午9:24
"""

import sys
from sklearn.cluster import KMeans
from sklearn.externals import joblib

sys.path.append('..')


def train_cluster_model(cluster_model_path,data):
    model = KMeans(n_clusters=20, verbose=True, n_jobs=8)
    model.fit(data)
    joblib.dump(model, cluster_model_path)
    return model

def load_cluster_model(cluster_model_path):
    model = joblib.load(cluster_model_path)
    return model

def cluster_model_predict(model,data):
    return model.predict(data)
