from __future__ import division

import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import math
import time


def create_sample(points=100, features=2):
  dim = (points,features)
  pt1 = np.random.normal(1, 0.2, dim)
  pt2 = np.random.normal(2, 0.5, (300,2))
  pt3 = np.random.normal(3, 0.3, dim)
  pt2[:,0] += 1
  pt3[:,0] -= 0.5
  arr = np.concatenate((pt1, pt2, pt3))
  return np.array(arr, dtype=np.float32)

def plotit(x, centroids, c):
  plt.close()
  fig = plt.figure()
  fig.suptitle("K means", fontsize=15)
  plt.scatter(centroids[:,0].T, centroids[:,1].T, s=200.0, c='r', marker='o', zorder=100)
  plt.plot(x[:,0], x[:,1], 'ko', zorder=-1)
  plt.xlabel('x1')
  plt.ylabel('x2')
  plt.show(block=False)
  time.sleep(1)

def initial_outlier():
  return np.array([
    [4.0,0.0],
    [0.0,4.0],
    [4.0,4.0],
    [4.1,4.1], 
    [0.0,0.0]
  ])

def initial_centroids(x, k):
  space = list(range(0, x.shape[0]))
  s = random.sample(space, k-2)
  return x[(np.array(s))]

def assign_centroids(c, x, centroids, m):
  similarity = lambda a,b,i: ((1 / m) * sim(a, b), i)
  for i in range(0, m):
    calc = [similarity(x[i], ce, idx) for idx, ce in enumerate(centroids)]
    c[i] = min(calc, key=lambda x: x[0])[1]
  return c

def sim(a, b):
  sum_squared = sum( (a - b)**2 )
  return (sum_squared**2)

def move_centroids(centroids, x, c, k):
  for j in range(0, k):
    cids = [idx for idx, i in enumerate(c) if i == j]
    try:
      centroids[j] = (1 / len(cids)) * sum(x[[cids]])
    except:
      continue
  return centroids

def converged(j_history, d):
  if d > 2:
    return (j_history[d-1] - j_history[d]) <= 0.000001
  return False

def kmeans(x, k, num_iter=25, printOut=False):
  j_history = []
  (m, n) = x.shape
  c = np.zeros((m, 1), dtype=np.float32)
  centroids = initial_centroids(x, k)
    
  if printOut: plotit(x, centroids, c)

  for d in range(0, num_iter):
    j_history.append( J(x, centroids, c) )
    print( j_history[d] )

    c = assign_centroids(c, x, centroids, m)
    centroids = move_centroids(centroids, x, c, k)

    if printOut: plotit(x, centroids, c)
    if converged(j_history, d):
      print("converged in round {}".format(d))
      break

  if printOut: plt.show()
  return centroids, c

def J(x, centroids, c):
  inter, variation = 0, 0
  (m, n) = x.shape
  for idx, cidx in enumerate(c):
    cidx = int(cidx)
    inter = inter + sim(x[idx], centroids[cidx])
    variation = variation + centroids[cidx]**2
  inter = (1 / m) * inter
  variation = (1 / m) * sum(variation)
  return (inter / variation)


def find_global_optimum(x, k, f, rounds=10):
  centroids, c = kmeans(x, k)
  cost, idx = J(x, centroids, c), 1

  for i in range(1, rounds):
    centroids_, c_ = kmeans(x, k)
    new_J = J(x, centroids_, c_)

    if f(new_J, cost):
      cost, idx = new_J, i + 1
      centroids, c = centroids_, c_

  return idx, cost, centroids, c


k, x = 5, create_sample()
by_min = lambda a,b: a <= b
idx, cost, centroids, c = find_global_optimum(x, k, by_min)

