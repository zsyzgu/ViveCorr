import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from hmmlearn import hmm
from motions_io import load_motions
import pandas as pd
from pandas.tools.plotting import lag_plot
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import math
import scipy.signal as signal

def movement_feature(movement):
      length = 24
      #72 dim --> 24 dim, velocity of each pos
      #27 dim(head and two hands) --> 9 dim
      #movement = movement[:, 0 : length] # (x, y, z) of three controllers
      movement = [np.array([math.sqrt(sum(movement[i][j : j + 3] ** 2)) for j in range(0, length, 3)]) for i in range(len(movement))] # velocity of three controllers
      movement = pd.rolling_mean(np.array(movement), 5)[4 :] #rolling mean (window = 5)
      #movement = signal.medfilt(movement, 5) #med filt (window = 5)
      return movement

def dtw(movement_0, movement_1):
      n = len(movement_0)
      m = len(movement_1)
      f = -np.ones((n, m))
      path = [[(0, 0) for c in range(m)] for r in range(n)]
      #dist = [[math.sqrt(sum((movement_0[r] - movement_1[c]) ** 2)) for c in range(m)] for r in range(n)]
      f[0][0] = 0
      for r in range(n):
            for c in range(m):
                  if (r == 0 and c == 0):
                        continue
                  if (r - 1 >= 0 and (f[r][c] == -1 or f[r][c] > f[r - 1][c])):
                        f[r][c] = f[r - 1][c]
                        path[r][c] = (r - 1, c)
                  if (c - 1 >= 0 and (f[r][c] == -1 or f[r][c] > f[r][c - 1])):
                        f[r][c] = f[r][c - 1]
                        path[r][c] = (r, c - 1)
                  if (r - 1 >= 0 and c - 1 >= 0 and (f[r][c] == -1 or f[r][c] > f[r - 1][c - 1])):
                        f[r][c] = f[r - 1][c - 1]
                        path[r][c] = (r - 1, c - 1)
                  f[r][c] += math.sqrt(sum((movement_0[r] - movement_1[c]) ** 2))
      (r, c) = (n - 1, m - 1)
      ixy = []
      while ((r, c) != (0, 0)):
            ixy.append((r, c))
            (r, c) = path[r][c]
      ixy.append((0, 0))
      ixy.reverse()
      return ixy

def alignment(x, y, ixy):
      n = len(ixy)
      x = np.array([x[ixy[i][0]] for i in range(n)])
      y = np.array([y[ixy[i][1]] for i in range(n)])
      return x, y

def get_residuals(movement):
      n = len(movement)

      model = VAR(movement)
      results = model.fit(1)
      predict_movement = np.concatenate((movement[: 2], [results.forecast(movement[i - 2 : i], 1)[0] for i in range(2, n)]))
      residuals = predict_movement - movement

      pca_n = 5
      pca = PCA(n_components = pca_n)
      residuals_pca = pca.fit_transform(residuals)

      #residuals_reprojected = pca.inverse_transform(residuals_pca)
      #loss = np.mean([math.sqrt(mean_squared_error(residuals[i], residuals_reprojected[i])) for i in range(n)])
      #print loss

      residuals = residuals_pca.T # zero mean unit variance
      residuals = np.array([(residuals[i] - np.mean(residuals[i])) / np.std(residuals[i]) for i in range(pca_n)]).T
      return residuals

def caln_entropy(residuals):
      n = len(residuals)
      entropy = (n / 2.0) * math.log(2 * math.pi * math.exp(1) * np.mean([math.sqrt(sum(residuals[i] ** 2)) for i in range(n)]), 2)
      return entropy

def caln_mutual_info(residuals_0, residuals_1):
      mutual_info = 0
      m = np.shape(residuals_0)[1]
      n = min(len(residuals_0), len(residuals_1))
      for i in range(m):
            f0 = residuals_0[: n, i]
            f1 = residuals_1[: n, i]
            p = np.corrcoef(f0, f1)[0][1]
            mutual_info = mutual_info - (n / 2.0) * math.log(1 - p ** 2 , 2) - math.log(math.exp(1), 2) / 2
      return mutual_info

data = load_motions('data/gyz_vec_0.txt') #inner_kick_left & inner_kick_right

movement_0 = movement_feature(data['inner_kick_left'][0])
movement_1 = movement_feature(data['inner_kick_left'][1])
ixy = dtw(movement_0, movement_1)
residuals_0 = get_residuals(movement_0)
residuals_1 = get_residuals(movement_1)

residuals_0, residuals_1 = alignment(residuals_0, residuals_1, ixy)

mutual_info = caln_mutual_info(residuals_0, residuals_1)
print mutual_info

'''
for key in data:
      print key

      for i in range(0, len(data[key]), 2):
            movement_0 = movement_feature(data[key][i + 0])
            movement_1 = movement_feature(data[key][i + 1])
            ixy = dtw(movement_0, movement_1)
            residuals_0 = get_residuals(movement_0)
            residuals_1 = get_residuals(movement_1)

            residuals_0, residuals_1 = alignment(residuals_0, residuals_1, ixy)

            mutual_info = caln_mutual_info(residuals_0, residuals_1)
            print mutual_info

      print ""
'''
