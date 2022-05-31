import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.decomposition import PCA
import codecs
corpus = []
with codecs.open("./new_symptom.csv","r","utf-8") as f:
    corpus = f.readlines()

filename = ["text_vectors_afterNormalize"]
components = [384 , 256 , 768]
poolings = ['first-last-avg', 'last-avg', 'cls', 'pooler']
for c in components:
    for pooling in poolings:
        for file in filename:
            vectors =  np.loadtxt("./vectors/%s/%s/%s.txt" %(str(c), pooling ,file))
            pca = PCA(n_components=2)
            vectors_ = pca.fit_transform(vectors)   #降维到二维
            r = 0.02
            db = DBSCAN(eps=r)
            y_ = db.fit_predict(vectors_)       #聚类
            label = db.labels_
            # plt.rcParams['font.sans-serif'] = ['FangSong']
            # plt.scatter(vectors_[:,0],vectors_[:, 1],s = 3 , c=y_)   #将点画在图上
            for i in range(len(corpus)):    #给每个点进行标注
                # plt.annotate(s=corpus[i], xy=(vectors_[:, 0][i], vectors_[:, 1][i]),
                #              xytext=(vectors_[:, 0][i] + 0.1, vectors_[:, 1][i] + 0.1))
                f = open('./dbscan/%s/%s/%s/label_%s.txt'  %(str(c), pooling , file ,str(label[i])) , 'a')
                f.write(corpus[i])
                f.close
            plt.rcParams['font.sans-serif'] = ['FangSong']
            plt.scatter(vectors_[:,0],vectors_[:, 1],s = 3 , c=y_)   #将点画在图上
            # for i in range(len(corpus)):    #给每个点进行标注
            #     # plt.annotate(s=corpus[i], xy=(vectors_[:, 0][i], vectors_[:, 1][i]),
            #     #              xytext=(vectors_[:, 0][i] + 0.1, vectors_[:, 1][i] + 0.1))
            #     plt.scatter(vectors_[:, 0][i], vectors_[:, 1][i], s = 3)
            plt.savefig(fname = ("./dbscan/%s_%s_%s_%s.jpg" %(str(c), pooling, r ,file)))
            plt.cla()