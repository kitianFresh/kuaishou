#coding=utf-8

"""
@author: coffee
@time: 18-6-29 下午2:39
"""
import pqkmeans
import pickle



def train(data,cluster_nums,model_path):
    assert data[0].shape[0] > 4
    encoder = pqkmeans.encoder.PQEncoder(num_subdim=4,Ks=256)
    encoder.fit(data)
    X_pqcode = encoder.transform(data)
    kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder,k=cluster_nums)
    kmeans.fit(X_pqcode)
    pickle.dump(encoder,open(model_path + '_encoder' + str(cluster_nums) + '.pkl','wb'))
    pickle.dump(kmeans,open(model_path + '_kmeans' + str(cluster_nums) + '.pkl','wb'))
    clustered = kmeans.predict(X_pqcode)
    return clustered

def predict(data,model_path, cluster_nums=50):
    encoder = pickle.load(open(model_path + '_encoder' + str(cluster_nums) + '.pkl','rb'))
    kmeans = pickle.load(open(model_path + '_kmeans' + str(cluster_nums) + '.pkl','rb'))
    X_pqcode = encoder.transform(data)
    clustered = kmeans.predict(X_pqcode)
    return clustered


