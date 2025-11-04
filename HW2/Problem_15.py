import numpy as np
# Problem 15. Calculate the Minkowski distance matrix (k x n) between rows 
# of 'a' (n x m) and 'b' (k x m) using parameter 'p'.
# Formula: D(x, y) = (sum(|x_i - y_i|^p))^(1/p)
def dist_mat(a, b, p):
    if a.ndim != 2 or b.ndim != 2:
        return "DimensionError"
    if a.shape[1] != b.shape[1]:
        return "DimensionError"

    n, k = a.shape[0], b.shape[0]
    c = np.zeros((k, n))

    for i in range(k):
        for j in range(n):
            c[i, j] = np.sum(np.abs(a[j] - b[i]) ** p) ** (1 / p)

    return c

# Do not modify this cell
a = np.array([[-15, 1.5],
              [10, 7],
              [1, 35],
              [4, -5]])
b = np.array([[1, 16],
              [-1, 0],
              [-1, 50]])
p = 3
assert np.all(np.round(dist_mat(a, b, p)) ==
              [[19, 11, 19, 21],
               [14, 12, 35,  6],
               [49, 43, 15, 55]])

a = np.array([[-15, 1.5],
              [10, 7],
              [1, 35],
              [4, -5]])
b = np.array([[1, 16, 1],
              [-1, 0, 1]])
p = 3
assert dist_mat(a, b, p) == "DimensionError"