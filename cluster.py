import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import codecs
corpus = []
with codecs.open("./new_test.csv","r","utf-8") as f:
    corpus = f.readlines()
km = KMeans(n_clusters=2)
pca = PCA(n_components=2)

vectors =  np.loadtxt("text_vectors.txt")
vectors_ = pca.fit_transform(vectors)   #降维到二维
y_ = km.fit_predict(vectors_)       #聚类
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.scatter(vectors_[:,0],vectors_[:, 1],c=y_)   #将点画在图上
for i in range(len(corpus)):    #给每个点进行标注
    plt.annotate(s=corpus[i], xy=(vectors_[:, 0][i], vectors_[:, 1][i]),
                 xytext=(vectors_[:, 0][i] + 0.1, vectors_[:, 1][i] + 0.1))
plt.savefig()