from __future__ import division

import numpy as np
import sklearn.decomposition as deco
import random
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import math
import time
import six


# Helper
def create_sample(num_docs, features):
  splitted = num_docs / 5
  a = 5 * np.random.random_sample((splitted, features))
  b = -5 * np.random.random_sample((splitted, features))
  c = np.random.normal(5, 4, (splitted, features))
  d = np.random.normal(-5, 4, (splitted, features))
  e = np.random.normal(0, 4, (splitted, features))
  return np.concatenate((a,b,c,d,e))


# Algorithm
def distance(x, centroids):
  return np.linalg.norm(x - centroids, axis=1)**2

def cost_func(x, centroids, c):
  return (1 / x.shape[0]) * np.sum( distance(x, centroids[c.flatten()]) )

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
    try:
      centroids[j] = (1 / len(cids)) * np.sum(x[cids], axis=0)
    except:
      pass
  return centroids.reshape(centroids.shape[0], -1)

def converged(j_history, i):
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

  print("cost:", j_history[-1], "k:", k )

  return centroids, c, j_history


def find_optimum(x, k=2, k_iter=10, num_iter=5):
  m = x.shape[0]
  centroids, c, j_history = None, None, None
  cost = float("inf")
  k_best, in_iter = k, 1

  iter_round = 0
  for i in range(0, num_iter):
    for k_off in range(0, k_iter):
      k_next = k + k_off
      centroids_, c_, j_history_ = kmeans(x, k_next)
      new_J = cost_func(x, centroids_, c_)

      if new_J < cost:
        print( "{}. new cost: {}, k: {}".format(iter_round + k_off, new_J, k_next) )
        in_iter = (i * k_iter) + (k_off + 1)
        cost, k_best = new_J, k_next
        centroids, c, j_history = centroids_, c_, j_history_

    iter_round = ((i + 1) * k_iter)

  return in_iter, cost, centroids, c


def reduce_dimensions(x, dims=3):
  x = (x - np.mean(x, 0)) / np.std(x, 0)
  pca = deco.PCA(dims)
  y = pca.fit(x).transform(x)
  print("Reducing columns of x {} by {} dims to {}".format(x.shape, dims, y.shape))
  return y


# Print out
def get_colors(k):
  ugly_colors = set(["white", "black", "cyan", "magenta", "pink"])
  colors_ = set([x[0] for x in list(six.iteritems(colors.cnames))]) - ugly_colors
  return list(set(random.sample(colors_, k)))

def cluster_plot_3d(x, centroids, c, k):
  fig = plt.figure()
  fig.canvas.set_window_title("K means")
  ax = fig.add_subplot(111, projection='3d')
  fig.suptitle("k = {}".format(k), fontsize=12)
  point_colors = get_colors(k)

  for i, cidx in enumerate(c):
    ax.scatter(x[i,0], x[i,1], x[i,2], 
      c=point_colors[cidx], marker='o', zorder=-1
    )
  for i in range(0, k):
    ax.scatter(centroids[i,0], centroids[i,1], centroids[i,2], 
      s=180.0, c=point_colors[i], marker='o', lw=2, zorder=100
    )
  plt.show()

def cluster_plot_2d(x, centroids, c, k):  
  fig = plt.figure()
  fig.canvas.set_window_title("K means")
  fig.suptitle("k = {}".format(k), fontsize=12)
  point_colors = get_colors(k)

  for i, cidx in enumerate(c):
    plt.scatter(x[i,0], x[i,1], 
      c=point_colors[cidx], marker='o', zorder=-1
    )
  for i in range(0, k):
    plt.scatter(centroids[i,0], centroids[i,1], 
      s=180.0, c=point_colors[i], marker='o', lw=2, zorder=100
    )
  
  plt.show()


if __name__ == "__main__":
  if False:
    x = create_sample(800, 10)
    x = reduce_dimensions(x, 2)
    in_iter, cost, centroids, c = find_optimum(x, k=1, k_iter=20, num_iter=3)
    k_best = centroids.shape[0]
    print( "iteration: {}, cost: {}, k: {}".format( in_iter, cost, k_best ) )
    cluster_plot_2d(x, centroids, c, k_best)

  else:
    k = 10
    x = create_sample(500, 10)
    x = reduce_dimensions(x, 2)
    centroids, c, j_history = kmeans(x, k)
    cluster_plot_2d(x, centroids, c, k)


