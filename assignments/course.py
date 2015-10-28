from __future__ import division

import numpy as np


def peek(data, n=3):
  print("\n".join(data.split("\n")[0:n]))

def get_data(folder, file_, tokenizer=None, conversion=None):
  if not tokenizer:
    tokenizer = lambda data: [tuple(l.strip().split(",")) for l in data.split("\n") if l.strip()]
  if not conversion:
    conversion = lambda a: np.array(a, dtype=np.double)
  base = "/Users/sacry/dev/projects/stanford_ml/assignments"
  open_path = "{}/{}/{}".format(base, folder, file_)
  with open(open_path, "r+") as f:
    data = f.read()
    return conversion(tokenizer(data))

# Excercise 1
def get_xyt(d):
  o = np.ones(len(d))
  x = np.column_stack((o, d[:,0]))
  y = d[:,1]
  theta = np.zeros((2,1))
  return x, y, theta

def computeCost(x, y, theta):
  vectorized = theta.T * x
  h0 = vectorized.sum(axis=1) - y.T.flatten()
  multiplier = 1 / (2 * len(y))
  return multiplier * sum(h0**2)

d = get_data("ex1", "ex1data1.txt")
x, y, theta = get_xyt(d)
print(computeCost(x, y, theta))

x = np.array([[1,2],[1,3],[1,4],[1,5]])
y = np.array([[7],[6],[5],[4]])
theta = np.array([[0.1],[0.2]])
print(computeCost(x, y, theta))

# Excercise 1 Optional
# d = get_data("ex1", "ex1data2.txt")



