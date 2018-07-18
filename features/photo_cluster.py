#coding=utf-8

"""
@author: coffee
@time: 18-7-1 上午11:51
"""
import numpy as np
import bloscpack as bp
import sys
sys.path.append('..')
from common import cluster
import os
import pandas as pd



def data_transform_store(visual_path,matrix_store_path,photo_id_path):
    photo = np.load(visual_path)
    keys = photo.keys()
    vector = np.zeros([len(keys)-1, 2048])
    photo_id = [0 for j in range(len(keys)-1)]
    print('start transform single file to matrix')
    for i in range(len(keys)):
        if i == 0:
            continue
        photo_id[i] = int(keys[i].split('/')[1])
        vec = photo[keys[i]]
        vector[i-1, :] = vec
    '''
    for index,name in enumerate(os.listdir(visual_path)):
        vector_path = os.path.join(visual_path,name)
        photo_id.append(int(name))
        if index == 0:
            vector = np.load(vector_path)
        else:
            vec = np.load(vector_path)
            vector = np.vstack([vector,vec])
    '''
    photo_id = np.array(photo_id)
    print('start store visual matrix and photo id')
    bp.pack_ndarray_file(vector,matrix_store_path)
    bp.pack_ndarray_file(photo_id,photo_id_path)


def sample_data_transform_store(visual_path,matrix_store_path,photo_id_path):
    vector = np.empty(([1,2048]))
    photo_ids = []
    print('start transform single file to matrix')

    vector_file = open(visual_path,'r')
    for index ,vector_path in enumerate(vector_file):
        vector_path = vector_path.replace('\n','')
        photo_id = vector_path.split('/')[-1]
        photo_ids.append(int(photo_id))
        if index == 0:
            vector = np.load(vector_path)
        else:
            vec = np.load(vector_path)
            vector = np.vstack([vector,vec])
    photo_ids = np.array(photo_ids)
    print('start store visual matrix and photo id')
    bp.pack_ndarray_file(vector,matrix_store_path)
    bp.pack_ndarray_file(photo_ids,photo_id_path)

def load_data(matrix_path,photo_id_path):
    matrix = bp.unpack_ndarray_file(matrix_path)
    photo_id = bp.unpack_ndarray_file(photo_id_path)
    return matrix,photo_id


def train_cluster_model(matrix,photo_id,cluster_model_path,cluster_nums):
    clustered = cluster.train(matrix, cluster_nums, cluster_model_path)
    photo_df = {'photo_id':photo_id,'photo_cluster_label':clustered}
    photo_df = pd.DataFrame(photo_df)
    return photo_df

def cluster_model_predict(cluster_model_path,matrix,photo_id):
    clustered = cluster.predict(matrix,cluster_model_path)
    photo_df = {'photo_id':photo_id,'photo_cluster_label':clustered}
    photo_df = pd.DataFrame(photo_df)
    return photo_df
