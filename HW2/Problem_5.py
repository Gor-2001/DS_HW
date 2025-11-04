import numpy as np
# Problem 5. Pad a 2D numpy array 'a' of size n x m with 0s 
# so that the dimensions become (n + 2*n1) x (m + 2*m1).
def padding(a, n1, m1):
    return np.pad(a, pad_width=((n1, n1), (m1, m1)), mode='constant', constant_values=0)

# Do not modify this cell
assert np.all(padding(np.array([[1, 1], [1, 1]]), 1, 2) ==
              [[0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0]])

assert np.all(padding(np.array([[1, 2], [1, 1]]), 0, 2) ==
              [[0, 0, 1, 2, 0, 0],
               [0, 0, 1, 1, 0, 0]])

assert np.all(padding(np.array([[9]]), 2, 2) ==
              [[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 9, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]])