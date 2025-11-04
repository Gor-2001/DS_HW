import numpy as np
# Problem 13. Find the rows of a matrix 'a' whose sum of elements 
# is in the range [100, 200).
def between_100_200(a):
    b = np.array(np.sum(a, axis=1))
    return a[(b < 200) & (b >= 100)]

# Do not modify this cell
a = np.array([[1, 1, 2],
              [0, 108, 3],
              [1, 3, 65],
              [50, 35, 5],
              [5, 83, 110],
              [98, 99, 10],
              [8, 9, 103],
              [9, 23, 15]])
assert np.all(between_100_200(a) ==
              [[0, 108,   3],
               [5,  83, 110],
               [8,   9, 103]])

a = np.array([[1, 1, 2],
              [0, -108, 3],
              [1, 3, 65],
              [50, 35, 5]])
assert len(between_100_200(a)) == 0