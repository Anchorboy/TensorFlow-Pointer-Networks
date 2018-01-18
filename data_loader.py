import itertools
import threading
import numpy as np
from tqdm import trange, tqdm
from collections import namedtuple

import tensorflow as tf

TSP = namedtuple('TSP', ['x', 'y', 'name'])

def length(x, y):
  return np.linalg.norm(np.asarray(x) - np.asarray(y))

# https://gist.github.com/mlalevic/6222750
def solve_tsp_dynamic(points):
  #calc all lengths
  all_distances = [[length(x,y) for y in points] for x in points]
  #initial value - just distance from 0 to every other point + keep the track of edges
  A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(all_distances[0][1:])}
  cnt = len(points)
  for m in range(2, cnt):
    B = {}
    for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
      for j in S - {0}:
        B[(S, j)] = min( [(A[(S-{j},k)][0] + all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
    A = B
  res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
  return np.asarray(res[1]) + 1 # 0 for padding

def generate_one_example(n_nodes, rng):
  nodes = rng.rand(n_nodes, 2).astype(np.float32)
  solutions = solve_tsp_dynamic(nodes)
  return nodes, solutions

def read_paper_dataset(path):
  i = 0
  x, y = [], []
  tf.logging.info("Read dataset {} which is used in the paper..".format(path))
  with open(path) as f:
    for l in tqdm(f):
      if i == 1000:
          break
      i += 1
      inputs, outputs = l.split(' output ')
      x.append(np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]))
      y.append(np.array(outputs.split(), dtype=np.int32)[:-1]) # skip the last one
  return x, y

class DataLoader(object):
  def __init__(self, config):
    self.config = config

  def read_data(self, path, name='tsp'):
    x_train_list, y_train_list = read_paper_dataset(path + '.txt')
    x_test_list, y_test_list = read_paper_dataset(path + '_test.txt')

    x_train = np.zeros([len(x_train_list), self.config.max_length, self.config.encode_size], dtype=np.float32)
    y_train = np.zeros([len(y_train_list), self.config.max_length+1], dtype=np.int32)

    x_test = np.zeros([len(x_test_list), self.config.max_length, self.config.encode_size], dtype=np.float32)
    y_test = np.zeros([len(y_test_list), self.config.max_length+1], dtype=np.int32)

    for idx, (nodes, res) in enumerate(tqdm(zip(x_train_list, y_train_list))):
      x_train[idx,:len(nodes)] = nodes
      y_train[idx,1:len(res)+1] = res

    for idx, (nodes, res) in enumerate(tqdm(zip(x_test_list, y_test_list))):
      x_test[idx,:len(nodes)] = nodes
      y_test[idx,1:len(res)+1] = res

    data = {}
    tf.logging.info("Update [{}] data with {} used in the paper".format(name, path))
    data['train'] = TSP(x=x_train, y=y_train, name=name)
    data['test'] = TSP(x=x_test, y=y_test, name=name)
    self.data = data

  def get_train_data(self, shuffle=True):
      data = self.data['train']
      assert len(data[0]) == len(data[1])
      dec_inputs = []
      for i in range(len(data[0])):
          tmp = []
          j = data[0][i]
          for k in data[1][i]:
              tmp += [j[k-1]] if k else [[0., 0.]]
          dec_inputs += [tmp]
      dec_inputs = np.asarray(dec_inputs, dtype=np.float32) 
      p = np.arange(len(data[0]))
      if shuffle:
          p = np.random.permutation(len(data[0]))
      labels = np.zeros_like(data[1])
      labels[:, :-1] = data[1][:, 1:]
      return data[0], dec_inputs[p], labels

  def get_test_data(self):
      data = self.data['test']
      assert len(data[0]) == len(data[1])
      dec_inputs = []
      for i in range(len(data[0])):
          tmp = []
          j = data[0][i]
          for k in data[1][i]:
              tmp += [j[k-1]] if k else [[0., 0.]]
              dec_inputs += [tmp]
      dec_inputs = np.asarray(dec_inputs, dtype=np.float32) 
      labels = np.zeros_like(data[1])
      labels[:, :-1] = data[1][:, 1:]
      shp = data[1].shape
      # dec_inputs = np.zeros((shp[0], shp[1], 2))
      return data[0], dec_inputs, labels
