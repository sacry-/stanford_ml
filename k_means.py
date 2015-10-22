from __future__ import division

import numpy as np
import random
import sys
import matplotlib.pyplot as plt


def initial_clusters(k, x):
  s = random.sample(list(range(0,x.shape[0])), k)
  return x[(np.array(s))]

def sim(a, b):
  return sum((a - b)**2)

def find_closest_cluster(ck, point):
  m = sim(point, ck[0])
  cluster_idx = 0
  for idx, cluster in enumerate(ck[1:]):
    s = sim(point, cluster)
    if s <= m:
      m = s
      cluster_idx = idx + 1
  return cluster_idx

def mean_of_points(c, x, j):
  cids = [idx for idx, i in enumerate(c) if i == j]
  return (1 / len(cids)) * sum(x[[cids]])

def k_means(k, x, num_iter=25):

  (m, n) = x.shape
  c = np.zeros((m, 1), dtype=np.float32)
  clusters = initial_clusters(k, x)

  for _ in range(0, num_iter):

    for i in range(0, m):
      c[i] = find_closest_cluster(clusters, x[i])

    for j in range(0, k):
      clusters[j] = mean_of_points(c, x, j) 

  return clusters, c


def create_sample(points=100, features=2):
  dim = (points,features)
  pt1 = np.random.normal(1, 0.2, dim)
  pt2 = np.random.normal(2, 0.5, (300,2))
  pt3 = np.random.normal(3, 0.3, dim)
  pt2[:,0] += 1
  pt3[:,0] -= 0.5
  arr = np.concatenate((pt1, pt2, pt3))
  return np.array(arr, dtype=np.float32)

def plotit(cl, x):
  plt.scatter(cl[:,0].T, cl[:,1].T, s=200.0, c='r', marker='o', zorder=100)
  plt.plot(x[:,0], x[:,1], 'ko', zorder=-1)
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.show()

# column: a[:,0]
k, x = 10, create_sample()
cl, c = k_means(k, x, 15)

plotit(cl, x)


