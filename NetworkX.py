# -*- coding: utf-8 -*-
"""
Created on Thu May  7 14:54:52 2020

@author: Hiro
"""

import math as m
import networkx as nx
import numpy as np
from numpy.linalg import eigh
from numpy.linalg import norm
from sklearn import datasets
from sklearn.cluster import KMeans, SpectralClustering
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
mpl.rcParams['figure.figsize'] = (8, 8)

data, target = datasets.make_moons(n_samples=100, noise=0.05)

def main():

    def k_nearest(data):

        weighted_adj = np.zeros((data.shape[0], data.shape[0]))
        for i in range(1,len(data)):
            for j in range(i):
                weighted_adj[j][i] = weighted_adj[i][j] = norm(data[i] - data[j])

        pos = {i: point for i, point in enumerate(data)}
        #print(wadj[0][0], wadj, pos)

        nearest = np.zeros((data.shape[0], data.shape[0]))
        k = 1
        copycat = []
        graph1 = nx.from_numpy_matrix(nearest)

        while nx.is_connected(graph1) == False:

            for i in range(len(data)):
                copy = deepcopy(weighted_adj[i])
                copy.sort()
                s = copy
                copy = copy[1:(k - len(copy))]

                for j in range(len(data)):
                    if weighted_adj[i][j] in copy:
                        nearest[i][j] = 1.0
                    else:
                        nearest[i][j] = 0.0

            graph1 = nx.from_numpy_matrix(nearest)
            k += 1

        plt.figure()
        plt.title("Graph of k-nearest neighbors")
        pos = {i: point for i, point in enumerate(data)}
        nx.draw_networkx_nodes(graph1, pos, node_size = 100, node_color='b')
        nx.draw_networkx_edges(graph1, pos, edge_color='y')

        return nearest

    def spectral_clustering(adj, data):

        def normalize_laplacian(L,D):
            L_norm = np.matmul(np.sqrt(D), L, np.sqrt(D))
            return L_norm

        clusters = 0
        graph = nx.from_numpy_matrix(adj)

        # Create the degree matrix
        D = np.diag([graph.degree()[i] for i in range(len(graph.degree()))])

        # Compute the Laplacian
        L = D - adj
        L = normalize_laplacian(L,D)
        eigvals, eigvects = eigh(L)

        # Obtain number of clusters
        tol = (np.sum(eigvals)/len(eigvals)) * 2

        for i in range(len(eigvals)):
            if eigvals[i] >= tol:
                clusters += 1
        if clusters == 0:
            clusters = 2

        # Run KMeans
        V = eigvects[:, :clusters]
        mns = KMeans(n_clusters=clusters, init='k-means++', n_init = 10).fit(V)
        labels = mns.labels_

        #Plots Sorted Data
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=labels)
        plt.title("Graph of sorted data")

        return labels

    graph = k_nearest(data)
    spectral_clustering(graph, data)