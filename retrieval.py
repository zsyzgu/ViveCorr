import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from hmmlearn import hmm
from motions_io import load_motions, Motion
import pandas as pd
from pandas.tools.plotting import lag_plot
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import math
import scipy.signal as signal
from sklearn.linear_model import LinearRegression 
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from sklearn.cross_decomposition import CCA

dim = 72
known = 27
unknown = 45
terms = 10

class Retrieval:
      def __init__(self):
            pass

      def set_train(self, train):
            self.train = train

      def set_origin(self, origin):
            self.origin = origin

      def dist(self, arr_0, arr_1):
            return math.sqrt(sum((arr_0 - arr_1) ** 2))

      def init_dtw(self):
            self.dtw = [-np.ones(len(self.origin[i].timestamp)) for i in range(len(self.origin))]
            self.match = np.zeros(len(self.origin), dtype = int)


      def fit_curr(self, X_pos):
            for i in range(len(self.origin)):
                  pos = self.origin[i].X_pos
                  dtw = self.dtw[i]
                  T = len(self.origin[i].timestamp)
                  new_dtw = -np.ones(T)
                  if (dtw[0] == -1):
                        new_dtw[0] = self.dist(pos[0], X_pos)
                  else:
                        new_dtw[0] = dtw[0] + self.dist(pos[0], X_pos)
                        for j in range(1, T):
                              new_dtw[j] = new_dtw[j - 1]
                              if (dtw[j - 1] != -1):
                                    new_dtw[j] = min(new_dtw[j], dtw[j - 1])
                              else:
                                    break
                              if (dtw[j] != -1):
                                    new_dtw[j] = min(new_dtw[j], dtw[j])
                              new_dtw[j] += self.dist(pos[j], X_pos)
                  self.dtw[i] = dtw = new_dtw
                  t = 0
                  for j in range(1, T):
                        if (dtw[j] != -1 and dtw[j] < dtw[t]):
                              t = j
                  self.match[i] = t

            t = int(np.mean(self.match))
            Y_pos = np.mean([self.train[i].Y_pos[t] for i in range(len(self.train))], axis = 0)

            return Y_pos

      def caln_error(self, test, predict):
            n = len(test)
            m = np.shape(test)[1]
            total = 0
            for i in range(n):
                  total += np.mean([math.sqrt(sum((test[i][j : j + 3] - predict[i][j : j + 3]) ** 2)) for j in range(0, m, 3)])
            error = total / n
            return error

      def fit(self, test):
            self.init_dtw()

            predict = Motion()
            T = len(test.timestamp)
            predict.Y_pos = [[] for i in range(T)]
            predict.Y_start = test.Y_start
            for i in range(T):
                  predict.Y_pos[i] = self.fit_curr(test.X_pos[i])

            error = self.caln_error(test.Y_pos, predict.Y_pos)
            print error

            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(test.Y_pos)
            plt.subplot(2, 1, 2)
            plt.plot(predict.Y_pos)
            plt.show()

motions = load_motions('data/gyz.txt')['knee_lift_right']
model = Retrieval()
model.set_train(motions[0 : 20])
model.set_origin(motions[20 : 25])
model.fit(motions[25])

#plt.savefig('pic/' + str(id) + '.jpg')
#test.output('pic/' + str(id) + '_gt.txt')
#predict.output('pic/' + str(id) + '.txt')
#predict.output('predict.txt')

'''
dim = 72
known = 27

class Retrieval:
      def __init__(self, motions):
            self.N = len(motions)
            self.train = motions
            self.dtw = [-np.ones(len(self.train[i].X_speed)) for i in range(self.N)]
            self.dtw_t = np.zeros(self.N, dtype = int)

      def dist(self, arr_0, arr_1):
            return math.sqrt(sum((arr_0 - arr_1) ** 2))

      def fit(self, X_pos, X_speed):
            for i in range(self.N):
                  speed = self.train[i].X_speed
                  dtw = self.dtw[i]
                  n = len(speed)
                  new_dtw = -np.ones(n)
                  if (dtw[0] == -1):
                        new_dtw[0] = self.dist(speed[0], X_speed)
                  else:
                        new_dtw[0] = dtw[0] + self.dist(speed[0], X_speed)
                        for j in range(1, n):
                              new_dtw[j] = new_dtw[j - 1]
                              if (dtw[j - 1] != -1):
                                    new_dtw[j] = min(new_dtw[j], dtw[j - 1])
                              else:
                                    break
                              if (dtw[j] != -1):
                                    new_dtw[j] = min(new_dtw[j], dtw[j])
                              new_dtw[j] += self.dist(speed[j], X_speed)
                  self.dtw[i] = dtw = new_dtw
                  t = 0
                  for j in range(1, n):
                        if (dtw[j] != -1 and dtw[j] < dtw[t]):
                              t = j
                  self.dtw_t[i] = t

            Y_pos = np.mean([self.train[i].Y_pos[self.dtw_t[i]] for i in range(self.N)], axis = 0)
            Y_speed = np.mean([self.train[i].Y_speed[self.dtw_t[i]] for i in range(self.N)], axis = 0)

            train_X = []
            train_Y = []
            for i in range(self.N):
                  t = self.dtw_t[i]
                  T = len(self.train[i].X_pos)
                  for dt in range(-4, 5):
                        if (t + dt >= 0 and t + dt < T):
                              train_X.append(self.train[i].X_pos[t + dt].copy())
                              train_Y.append(self.train[i].Y_pos[t + dt].copy())

            linreg = LinearRegression()
            reg = linreg.fit(train_X, train_Y)
            Y_pos = reg.predict([X_pos])[0]

            return Y_pos, Y_speed

def caln_dist(vec0, vec1):
      n = len(vec0)
      return np.mean([math.sqrt(sum((vec0[i : i + 3] - vec1[i : i + 3]) ** 2)) for i in range(0, n, 3)])

def caln_error(test, predict):
      n = len(test)
      return np.mean([caln_dist(test[i], predict[i]) for i in range(n)])

motions = load_motions('data/gyz731_vec.txt')['toe_kick_left']

test = motions[0]
predict = test.copy()
model = Retrieval(motions[1 :])

n = len(test.X_speed)
for i in range(n):
      print i
      predict.Y_pos[i], predict.Y_speed[i] = model.fit(test.X_pos[i], test.X_speed[i])
replay = np.array(predict.Y_pos)
for i in range(1, n):
      predict.Y_pos[i] = predict.Y_pos[i - 1] + predict.Y_speed[i - 1] * (predict.timestamp[i] - predict.timestamp[i - 1])
comb = np.array(replay)
for i in range(n):
      comb[i] = float(i) / n * replay[i] + (1 - float(i) / n) * predict.Y_pos[i]

print error1, error2, error3
plt.subplot(4, 1, 1)
plt.plot(test.Y_pos)
plt.subplot(4, 1, 2)
plt.plot(predict.Y_pos)
plt.subplot(4, 1, 3)
plt.plot(replay)
plt.subplot(4, 1, 4)
plt.plot(predict.Y_speed)
plt.show()
predict.output('predict.txt')
'''

'''
def dtw(movement_0, movement_1):
      distance, path = fastdtw(movement_0, movement_1, dist = euclidean)
      return path

data = load_motions('data/gyz731_vec.txt')['knee_lift_right']
n = len(data)
print n

ixy = [dtw(data[0], data[i]) for i in range(n)]

action = []

for t in range(T):
      transform = []
      transform.append(data[0][t].copy())
      for i in range(1, n):
            ty = 0
            while (ixy[i][ty][0] < t):
                  ty += 1
            ty = ixy[i][ty][1]
            transform.append(data[i][ty].copy())
            if (ty + 1 < T):
                  transform.append(data[i][ty + 1].copy())
            if (ty + 2 < T):
                  transform.append(data[i][ty + 2].copy())
            if (ty - 1 >= 0):
                  transform.append(data[i][ty - 1].copy())
            if (ty - 2 >= 0):
                  transform.append(data[i][ty - 2].copy())
      transform = np.array(transform)

      predict = [0 for i in range(dim)]
      X = transform[1 :, : known]
      Y = transform[1 :, known :]

      linreg = LinearRegression()
      model = linreg.fit(X, Y)
      predict[known :] = model.predict([transform[0, : known]])[0]
      #predict[known :] = [np.mean(transform[1 : n, i]) for i in range(known, m)]

      action.append(predict)

action = np.array(action)

plt.subplot(2, 2, 1)
plt.title('head and hands')
plt.xlim(0, T)
plt.plot(data[0][:, : known])
plt.subplot(2, 2, 2)
plt.title('legs (ground truth)')
plt.xlim(0, T)
plt.plot(data[0][:, known :])
plt.subplot(2, 2, 4)
plt.xlim(0, T)
plt.title('legs (predict)')
plt.plot(action[:, known :])
plt.show()
'''
