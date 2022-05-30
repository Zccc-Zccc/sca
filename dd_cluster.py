import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
import codecs
corpus = []
with codecs.open("./new_text.csv","r","utf-8") as f:
    corpus = f.readlines()
km = KMeans(n_clusters=10)
pca = PCA(n_components=2)

filename = ["text_vectors_bert","text_vectors_transform_384","text_vectors_afterNormalize_384","text_vectors_transform_256","text_vectors_afterNormalize_256"]

for file in filename:
    vectors =  np.loadtxt("%s.txt" %file)
    vectors_ = pca.fit_transform(vectors)   #降维到二维
    y_ = km.fit_predict(vectors_)       #聚类
    plt.rcParams['font.sans-serif'] = ['FangSong']
    plt.scatter(vectors_[:,0],vectors_[:, 1],s = 3 , c=y_)   #将点画在图上
    # for i in range(len(corpus)):    #给每个点进行标注
    #     # plt.annotate(s=corpus[i], xy=(vectors_[:, 0][i], vectors_[:, 1][i]),
    #     #              xytext=(vectors_[:, 0][i] + 0.1, vectors_[:, 1][i] + 0.1))
    #     plt.scatter(vectors_[:, 0][i], vectors_[:, 1][i], s = 3)
    plt.savefig(fname = ("5_%s" %file))
    plt.cla()
