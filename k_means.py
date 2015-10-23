from __future__ import division

import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import math
import pprint


def create_sample(points=100, features=2):
  dim = (points,features)
  pt1 = np.random.normal(1, 0.2, dim)
  pt2 = np.random.normal(2, 0.5, (300,2))
  pt3 = np.random.normal(3, 0.3, dim)
  pt2[:,0] += 1
  pt3[:,0] -= 0.5
  arr = np.concatenate((pt1, pt2, pt3))
  return np.array(arr, dtype=np.float32)

def plotit(cl, x, name="K means"):
  fig = plt.figure()
  fig.suptitle(name, fontsize=15)
  plt.scatter(cl[:,0].T, cl[:,1].T, s=200.0, c='r', marker='o', zorder=100)
  plt.plot(x[:,0], x[:,1], 'ko', zorder=-1)
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.show()


class KMeans():

  def __init__(self, k, x):
    self.k = k
    self.x = x
    self.num_iter = 10
    self.c = None
    self.centroids = None
    self.steps = {}

  def initial_centroids(self):
    space = list(range(0, self.x.shape[0]))
    s = random.sample(space, self.k)
    return self.x[(np.array(s))]

  def find_closest_cluster(self, point):
    m = self.sim( point, self.centroids[0] )
    cluster_idx = 0
    for idx, cluster in enumerate(self.centroids[1:]):
      s = self.sim(point, cluster)
      if s <= m:
        m = s
        cluster_idx = idx + 1
    return cluster_idx

  def run(self):
    (m, n) = self.x.shape
    self.c = np.zeros((m, 1), dtype=np.float32)
    self.centroids = self.initial_centroids()

    for iteration in range(0, self.num_iter):
      for i in range(0, m):
        self.c[i] = self.find_closest_cluster(x[i])

      for j in range(0, k):
        self.centroids[j] = self.mean_of_points(j) 

      self.steps[iteration] = self.centroids

    return self.centroids, self.c

  def mean_of_points(self, j):
    cids = [idx for idx, i in enumerate(self.c) if i == j]
    return (1 / len(cids)) * sum(self.x[[cids]])

  def sim(self, a, b):
    sum_squared = sum( (a - b)**2 )
    return math.sqrt(sum_squared)**2

  def J(self):
    inter = 0
    for idx, cidx in enumerate(self.c):
      cidx = int(cidx)
      inter = inter + self.sim(self.x[idx], self.centroids[cidx])
    return (1 / len(self.c)) * inter

  def __repr__(self):
    return "KMeans[k: {}, samples: {}, features: {}, num_iter: {}]".format(
      self.k, 
      self.x.shape[0], 
      self.x.shape[1], 
      self.num_iter
    )


def find_global_optimum(k, x, f, rounds=5):
  kmeans = KMeans(k, x)
  kmeans.run()
  cost, idx = kmeans.J(), 1
  best_kmeans = kmeans

  for i in range(1, rounds):
    for ki in range(k+1, k+5):
      kmeans = KMeans(k, x)
      kmeans.run()
      new_J = kmeans.J()

      if f(new_J, cost):
        cost, idx = new_J, i + 1
        best_kmeans = kmeans

  return idx, best_kmeans


k, x = 2, create_sample()
min_idx, kmeans = find_global_optimum(k, x, lambda a,b: a <= b)

print(str(kmeans), "Found in iteration: {}".format(min_idx))
pprint.pprint(kmeans.steps)
plotit(kmeans.centroids, kmeans.x, "Minimum")

