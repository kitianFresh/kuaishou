#coding=utf-8

"""
@author: coffee
@time: 18-6-26 下午7:42
"""

import gensim

def load_model(path):
    return gensim.models.Word2Vec.load(path)

def train_word2vec(data,path):
    model = gensim.models.Word2Vec(data,size=100,window=5,min_count=5,workers=8)
    model.save(path)
    return model

