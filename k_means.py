from __future__ import division

import numpy as np
import random
import sys
import matplotlib.pyplot as plt


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

def J(cl, c, x):
  accu = 0
  for idx, cidx in enumerate(c):
    cidx = int(cidx)
    inter = sum(x[idx] - cl[cidx])**2
    accu += inter
  return (1 / len(c)) * accu


def find_global_optimum(k, x, f, num_iter=1, samples=50):

  cl, c = k_means(k, x, num_iter)
  cost, idx = J(cl, c, x), 0

  for i in range(0, samples - 1):

    cl_, c_ = k_means(k, x, num_iter)
    new_J = J(cl_, c_, x)

    if f(new_J, cost):
      cost, idx = new_J, i
      cl, c = cl_, c_

  return idx, cost, cl, c


k, x = 5, create_sample()
min_idx, min_cost, cl_min, c_min = find_global_optimum(k, x, lambda a,b: a <= b, 1, 50)
max_idx, max_cost, cl_max, c_max = find_global_optimum(k, x, lambda a,b: a >= b, 1, 50)

print(min_idx, min_cost, cl_min.shape, c_min.shape)
print(max_idx, max_cost, cl_max.shape, c_max.shape)

plotit(cl_min, x, "Minimum")
plotit(cl_max, x, "Maximum")


