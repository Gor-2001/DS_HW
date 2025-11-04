import numpy as np
# Problem 4. One-hot encode a 1D numpy array 'a'.
# Let 'b' be the unique, sorted values of 'a' (size m). Create an n x m array 
# where the i-th row, j-th column is 1 if a[i] equals b[j], and 0 otherwise.
def convert(a):

    b = np.unique(a)
    a = a[:, None]
    return (a == b).astype(int)

# Do not modify this cell
assert np.all(convert(np.array([0, 0, 2, 3, 2, 4])) ==
              [[1, 0, 0, 0],
               [1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])

assert np.all(convert(np.array([-np.pi, 9])) ==
              [[1, 0],
               [0, 1]])

assert np.all(convert(np.array([-1, 1, 9, 10])) ==
              [[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

assert np.all(convert(np.array([np.sqrt(2), np.sqrt(2), np.sqrt(2)])) ==
              [[1],
               [1],
               [1]])