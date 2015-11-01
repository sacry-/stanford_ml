from __future__ import division

import numpy as np


class DataImport():

  def __init__(self, path_to):
    self.path_to = path_to
    self.raw_data = self.__read__(self.path_to)

    self.__tokenizer = None
    self.__converter = None

  def transform(self):
    tokens = self.__tokenize(self.raw_data)
    transformation = self.__convert(tokens)
    return transformation

  def peek(self, n=3):
    print( self.raw_data[0:n] )

  def set_tokenizer(self, tokenizer):
    self.__tokenizer = tokenizer

  def set_converter(self, converter):
    self.__converter = converter

  def __tokenize(self, raw_data):
    if not self.__tokenizer:
      return [l.strip().split(",") for l in raw_data.split("\n") if l.strip()]
    return self.__tokenizer(raw_data)

  def __convert(self, tokens):
    if not self.__converter:
      return np.array(tokens, dtype=np.double)
    return self.__converter(tokens)

  def __read__(self, path_to):
    base = "/Users/sacry/dev/projects/stanford_ml/assignments"
    open_path = "{}/{}".format(base, path_to)
    with open(open_path, "r+") as f:
      return f.read()



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

importer = DataImport("ex1/ex1data1.txt")
d = importer.transform()
x, y, theta = get_xyt(d)
print(computeCost(x, y, theta))

x = np.array([[1,2],[1,3],[1,4],[1,5]])
y = np.array([[7],[6],[5],[4]])
theta = np.array([[0.1],[0.2]])
print(computeCost(x, y, theta))

# Excercise 1 Optional
# d = get_data("ex1", "ex1data2.txt")



