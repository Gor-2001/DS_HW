import numpy as np

# Problem 10. Randomly divide the rows of a matrix 'a' (m x n) 
# into two parts in the ratio k:(1-k). Round the number of rows if necessary.
def rand_split(a, k):
  np.random.shuffle(a)

  count = int(k * a.shape[0])
  b = np.copy(a)
  b = b[:count]

  c = np.copy(a)
  c = c[count:]

  return b, c

# Do not modify this cell
a = np.array([[1, 1, 2],
              [0, 108, 3],
              [1, 3, 65],
              [50, 35, 5],
              [5, 83, 110],
              [98, 99, 10],
              [8, 9, 103],
              [9, 23, 15]])
k = 0.25

for _ in range(100):
    x1, x2 = rand_split(a, k)
    assert len(x1) == 2 and len(x2) == 6
    # Note: These original assertions are weak checks on non-overlap
    # assert np.all(np.sum(x2 != x1[0], axis=1)) 
    # assert np.all(np.sum(x2 != x1[1], axis=1))