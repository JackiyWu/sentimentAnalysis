import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math

index_1 = 2
index_2 = 3

X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
print("X = ", X)
plt.scatter(X[:, index_1], X[:, index_2], marker='o')
plt.savefig('result/clustering/figures/test.jpg')
plt.show()

y_pred = SpectralClustering().fit_predict(X)
print("SpectralClustering's Calinski-Harabasz Score", metrics.calinski_harabasz_score(X, y_pred))
plt.scatter(X[:, index_1], X[:, index_2], c=y_pred)
plt.savefig('result/clustering/figures/test2.jpg')
plt.show()

y_pred_2 = KMeans().fit_predict(X)
print("KMeans' Calinski-Harabasz Score", metrics.calinski_harabasz_score(X, y_pred_2))
plt.scatter(X[:, index_1], X[:, index_2], c=y_pred_2)
plt.savefig('result/clustering/figures/test3.jpg')
plt.show()

best_para_calnski = ()
best_para_silhouette = ()
best_para_dbi = ()
# best_para_calnski = (3, 0.01)
# best_para_silhouette = (3, 0.01)
# best_para_dbi = (3, 0.01)
best_score_calnski = -1  # 越大越好
best_score_silhouette = -1  # 越大越好
best_score_dbi = 2  # 越小越好
for index1, gamma in enumerate((0.01, 0.1, 1, 10)):
    for index2, k in enumerate((3, 4, 5, 6)):

        y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)

        # Calinski-Harabasz的值越大，代表聚类效果越好
        calinski_score = metrics.calinski_harabasz_score(X, y_pred)
        if calinski_score > best_score_calnski:
            best_score_calnski = calinski_score
            best_para_calnski = (k, gamma)
        print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k, "score:", calinski_score)
        # Silhouette Coefficient值越大，代表聚类效果越好，取值范围为-1~1之间
        silhouette_score = metrics.silhouette_score(X, y_pred)
        if silhouette_score > best_score_silhouette:
            best_score_silhouette = silhouette_score
            best_para_silhouette = (k, gamma)
        print("Silhouette Coefficient Score with gamma=", gamma, "n_clusters=", k, "score:", silhouette_score)
        # DBI的值越小，代表聚类效果越好，最小是0
        dbi_score = metrics.davies_bouldin_score(X, y_pred)
        if dbi_score < best_score_dbi:
            best_score_dbi = dbi_score
            best_para_dbi = (k, gamma)
        print("DBI Score with gamma=", gamma, "n_clusters=", k, "score:", metrics.davies_bouldin_score(X, y_pred))

        # y_pred_2 = KMeans(n_clusters=k).fit_predict(X)
        # print("KMeans' Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k, "score:", metrics.calinski_harabasz_score(X, y_pred_2))

print("best_para_calnski = ", best_para_calnski)
print("best_para_silhouette = ", best_para_silhouette)
print("best_para_dbi = ", best_para_dbi)
k = best_para_calnski[0]
gamma = best_para_calnski[1]

y_pred = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(X)
# print("y_pred = ", y_pred)
print("SpectralClustering's Calinski-Harabasz Score", metrics.calinski_harabasz_score(X, y_pred))
plt.scatter(X[:, index_1], X[:, index_2], c=y_pred)
plt.savefig('result/clustering/figures/test4.jpg')
plt.show()

y_pred_2 = KMeans(n_clusters=5).fit_predict(X)
# print("y_pred_2 = ", y_pred_2)
print("KMeans' Calinski-Harabasz Score", metrics.calinski_harabasz_score(X, y_pred_2))
plt.scatter(X[:, index_1], X[:, index_2], c=y_pred_2)
plt.savefig('result/clustering/figures/test5.jpg')
plt.show()

print(">>>this is the end of absa_spectralClustering.py...")

