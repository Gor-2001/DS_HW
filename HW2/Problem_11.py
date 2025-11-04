import numpy as np
# Problem 11. Randomly select 'k' part of the rows of 'a' (with replacement) 
# and 'q' part of their columns (without replacement).
def rand_rows_cols(a, k, q):

    n_rows, n_cols = a.shape
    rowCount = int(np.ceil(k * n_rows))
    columnCount = int(np.ceil(q * n_cols))
    
    rows_idx = np.random.choice(n_rows, size=rowCount, replace=True)
    cols_idx = np.random.choice(n_cols, size=columnCount, replace=False)
    return a[rows_idx][:, cols_idx]

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
q = 0.6

for _ in range(100):
    x1 = rand_rows_cols(a, k, q)
    assert x1.shape == (2, 2)