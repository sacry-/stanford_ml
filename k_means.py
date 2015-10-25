from __future__ import division

import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import math
import time


# Helper
def create_sample(points=100, features=2):
  dim = (points,features)
  pt1 = np.random.normal(1, 0.2, dim)
  pt2 = np.random.normal(2, 0.5, (300,2))
  pt3 = np.random.normal(3, 0.3, dim)
  pt2[:,0] += 1
  pt3[:,0] -= 0.5
  arr = np.concatenate((pt1, pt2, pt3))
  return np.array(arr, dtype=np.float32)

def plotit(x, centroids):
  plt.close()
  fig = plt.figure()
  fig.suptitle("K means", fontsize=15)
  plt.scatter(centroids[:,0].T, centroids[:,1].T, s=200.0, c='r', marker='o', zorder=100)
  plt.plot(x[:,0], x[:,1], 'ko', zorder=-1)
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.show(block=False)
  time.sleep(1)


# Algorithm
def initial_centroids(x, k):
  space = list(range(0, x.shape[0]))
  s = random.sample(space, k)
  return x[np.array(s)]

def assign_centroids(c, x, centroids, m):
  for i in range(0, m):
    c[i] = np.argmin((1 / m) * sim(x[i], centroids))
  return c

def sim(a, b):
  return np.sum( (a - b)**2, axis=1 )**2

def move_centroids(centroids, x, c, k):
  for j in range(0, k):
    (cids, _) = np.where(c == j)
    try:
      centroids[j] = (1 / len(cids)) * sum(x[[cids]])
    except:
      continue
  return centroids

def converged(j_history, d):
  if d > 2:
    diff = (j_history[d-1][0] - j_history[d][0])
    if diff < 0:
      return False
    return diff <= 0.000001
  return False

def kmeans(x, k, num_iter=25):
  j_history = []
  (m, n) = x.shape
  c = np.zeros((m, 1), dtype=np.float32)
  centroids = initial_centroids(x, k)

  for d in range(0, num_iter):
    cost = J(x, centroids, c)
    j_history.append( (cost, centroids) )
    print( cost )

    c = assign_centroids(c, x, centroids, m)
    centroids = move_centroids(centroids, x, c, k)

    if converged(j_history, d):
      print("converged in round {}".format(d))
      break

  return centroids, c, j_history

def J(x, centroids, c):
  inter, variation = 0, 0
  (m, n) = x.shape
  for idx, cidx in enumerate(c):
    cidx = int(cidx)
    inter = inter + sum( (x[idx] - centroids[cidx])**2 )**2
    variation = variation + centroids[cidx]**2
  inter = (1 / m) * inter
  variation = (1 / m) * sum(variation)
  return (inter / variation)


def find_global_optimum(x, f=lambda a,b: a <= b, k=1, k_iter=10, num_iter=5):
  centroids, c, j_history = kmeans(x, k)
  cost, idx = J(x, centroids, c), 1
  k_offset, k_best = 1, k

  for i in range(0, num_iter):
    for k_off in range(0, k_iter):
      centroids_, c_, j_history_ = kmeans(x, k + k_off)
      new_J = J(x, centroids_, c_)

      if f(new_J, cost):
        cost, idx, k_best, k_offset = new_J, i + 1, k + k_off, k_off
        centroids, c, j_history = centroids_, c_, j_history_

  return idx, k_offset, k_best, cost, centroids, c, j_history

def animate(j_history, x):
  for (_, cent) in j_history:
    plotit(x, cent)
  plt.show()


x = create_sample()
idx, k_offset, k, cost, centroids, c, j_history = find_global_optimum(x)
print(idx, k_offset, idx*k_offset, cost, k)
animate(j_history, x)

