import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import codecs
corpus = []
with codecs.open("./new_symptom.csv","r","utf-8") as f:
    corpus = f.readlines()
clusters = 10
km = KMeans(n_clusters=clusters)
pca = PCA(n_components=2)

filename = ["text_vectors_bert"]
poolings = ['first-last-avg', 'last-avg', 'cls', 'pooler']

for file in filename:
    for pooling in poolings:
        vectors =  np.loadtxt("./bert/%s/%s.txt" %(pooling , file))
        vectors_ = pca.fit_transform(vectors)   #降维到二维
        y_ = km.fit_predict(vectors_)       #聚类
        label = km.labels_
        # plt.rcParams['font.sans-serif'] = ['FangSong']
        
        for i in range(len(corpus)):    #给每个点进行标注
            # plt.annotate(s=corpus[i], xy=(vectors_[:, 0][i], vectors_[:, 1][i]),
            #              xytext=(vectors_[:, 0][i] + 0.1, vectors_[:, 1][i] + 0.1))
            f = open('./bert/%s/label_%s.txt'  %( pooling , str(label[i])), 'a')
            f.write(corpus[i])
            f.close
        plt.scatter(vectors_[:,0],vectors_[:, 1],s = 3 , c=y_)   #将点画在图上
        plt.savefig(fname= "%s_%s_bert" % (clusters,pooling))
        plt.cla()
