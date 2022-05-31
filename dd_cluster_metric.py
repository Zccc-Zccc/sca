import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import silhouette_score, silhouette_samples

import codecs


pca = PCA(n_components=2)

filename = ["text_vectors_transform","text_vectors_afterNormalize"]
components = [384 , 256 , 768]
poolings = ['first-last-avg', 'last-avg', 'cls', 'pooler']
c = 256

for pooling in poolings:
    for file in filename:
        vectors =  np.loadtxt("./%s/%s/%s.txt" %(str(c), pooling ,file))
        vectors_ = pca.fit_transform(vectors)   #降维到二维
        chScore = []
        scScore = []
        for i in range(21):
            if i < 5:
                continue
            km = KMeans(n_clusters=i)
            y_ = km.fit_predict(vectors_)       #聚类
            chScore.append(metrics.calinski_harabaz_score(vectors_,y_))
            scScore.append(silhouette_score(vectors_,km.labels_))
        with codecs.open("./%s/%s/%s/scScore.txt" %(str(c) , pooling , file) , 'a' , 'utf-8') as f:
            for i in range(len(scScore)):
                f.write("clusters_%s sc score : %s" %(str(i+5), scScore[i]))
        with codecs.open("./%s/%s/%s/chScore.txt" %(str(c) , pooling , file) , 'a' , 'utf-8') as f:
            for i in range(len(chScore)):
                f.write("clusters_%s ch score : %s" %(str(i+5), chScore[i]))
            
            
