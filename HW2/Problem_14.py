import numpy as np
# Problem 14. Get the L in the Cholesky decomposition of x (where x = L * L^T).
def Cholesky(x):
    return np.linalg.cholesky(x)

# Do not modify this cell
x = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
assert np.all(Cholesky(x) ==
              [[2.,  0.,  0.],
               [6.,  1.,  0.],
               [-8.,  5.,  3.]])