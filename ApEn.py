import numpy as np
import random

def ApEn(u, m, r):
      def max_dist(xi, xj):
            return max(abs(xi - xj))

      def phi(m):
            x = [u[i : i + m] for i in range(N - m + 1)]
            C = [len([1 for xj in x if max_dist(xi, xj) <= r]) / (N - m + 1.0) for xi in x]
            return (N - m + 1.0) ** (-1) * sum(np.log(C))

      N = len(u)

      return abs(phi(m) - phi(m + 1))

u = np.array([random.uniform(0, 10) for i in range(10)])
print u
print ApEn(u, 2, 3)
