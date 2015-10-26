from __future__ import division

import numpy as np
import random
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import math
import time
import six


# Helper
def create_sample(k, docs=400):
  nums = [x for x in range(1,4)]
  docs_per = docs / (k * 3)
  accu = np.zeros((1,2))

  def create_k_docs(accu):
    for i in range(0, k):
      variance = min(1.0 - random.random(), 0.5) + 0.5
      num = random.choice( nums )
      a = np.random.normal(num, variance, (docs_per, 2))
      accu = np.concatenate((accu, a))
    return accu

  accu = create_k_docs(accu)
  accu = create_k_docs(accu)

  for i in range(0, k*10):
    num = random.choice( nums )
    a = np.random.normal(4, 30, (5, 2))
    accu = np.concatenate((accu, a))

  return np.array(accu, dtype=np.float32)


# Algorithm
def distance(x, centroids):
  return np.log(np.sum((x - centroids)**2, axis=1))

def cost_func(x, centroids, c):
  return (1 / x.shape[0]) * np.sum( distance(x, centroids[c]) )

def initial_centroids(x, k):
  space = list(range(0, x.shape[0]))
  s = random.sample(space, k)
  return x[np.array(s)]

def assign_centroids(c, x, centroids):
  m = x.shape[0]
  for i in range(0, m):
    dist = (1 / m) * distance(x[i,:], centroids)
    c[i] = np.argmin(dist)
  return c

def move_centroids(centroids, x, c, k):
  for j in range(0, k):
    (cids, _) = np.where(c == j)
    centroids[j] = (1 / len(cids)) * np.sum(x[cids], axis=0)
  return centroids.reshape(centroids.shape[0], -1)

def converged(j_history, i):
  print( j_history[i] )
  if i > 3:
    diff = j_history[i-1] - j_history[i]
    return diff < 0.0000001 and not diff < 0
  return False

def kmeans(x, k, num_iter=40):
  j_history = []
  (m, _) = x.shape
  c = np.zeros((m, 1), dtype=np.int)
  centroids = initial_centroids(x, k)

  for i in range(0, num_iter):
    j_history.append( cost_func(x, centroids, c) )

    c = assign_centroids(c, x, centroids)
    centroids = move_centroids(centroids, x, c, k)

    if converged(j_history, i):
      print("converged at iteration: {}".format(i))
      break

  return centroids, c, j_history


# optima..
class FindOptimum():

  def __init__(self, x, k=2, k_iter=10, num_iter=5):
    self.x = x
    self.m = x.shape[0]
    self.k = k
    self.k_iter = k_iter
    self.num_iter = num_iter

  def search(self):
    self.centroids, self.c, self.j_history = None, None, None
    self.cost = np.sum(x)**2
    self.k_best, self.in_iter = self.k, 1

    iter_round = 0
    for i in range(0, self.num_iter):
      for k_off in range(0, self.k_iter):
        k_next = self.k + k_off
        centroids_, c_, j_history_ = kmeans(self.x, k_next)
        new_J = cost_func(self.x, centroids_, c_)

        print( "{}. cost: {}, k: {}".format(iter_round + k_off, new_J, k_next) )
        if new_J < self.cost:
          self.in_iter = (i * self.k_iter) + (k_off + 1)
          self.cost, self.k_best = new_J, k_next
          self.centroids, self.c, self.j_history = centroids_, c_, j_history_

      iter_round = ((i + 1) * self.k_iter)


def cluster_plot(x, centroids, c, k):
  fig = plt.figure()
  fig.suptitle("K means", fontsize=15)

  colors_ = set([x[0] for x in list(six.iteritems(colors.cnames))]) - set(["white", "black", "cyan", "magenta", "pink"])
  point_colors = set(random.sample(colors_, k))
  centroid_color = random.sample(colors_ - point_colors, 1).pop()
  point_colors = list(point_colors)

  for i, cidx in enumerate(c):
    plt.plot(x[i,0], x[i,1], point_colors[cidx], marker="o", zorder=-1)
  for i in range(0, k):
    plt.scatter(centroids[i,0], centroids[i,1], s=150.0, c=centroid_color, marker='o', zorder=100)
  plt.show()

if __name__ == "__main__":
  if False:
    x = create_sample(15)
    analyzer = FindOptimum(x, k=1, k_iter=5, num_iter=3)
    analyzer.search()
    print( "iteration: {}, cost: {}, k: {}".format( analyzer.in_iter, analyzer.cost, analyzer.k_best ) )
    cluster_plot(x, analyzer.centroids, analyzer.c, analyzer.k_best)

  else:
    k = 10
    x = create_sample(15)
    centroids, c, j_history = kmeans(x, k)
    cluster_plot(x, centroids, c, k)



