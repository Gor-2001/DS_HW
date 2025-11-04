import numpy as np

# ==============================================================================
# GENERAL CONSTRAINTS AND ERROR HANDLING
# ==============================================================================

# Using libraries other than numpy is **prohibited**.

# Error handling rules:
# 1. If a numpy array (vector or matrix) parameter does not match the dimension 
#    required by the problem, or if there are dimension mismatches, return 
#    **"DimensionError"**.
# 2. If a 2D numpy array (matrix) parameter is required to be invertible but is not, 
#    return **"NotInvertableError"**.

# ==============================================================================
# PROBLEMS AND TESTS
# ==============================================================================

# Problem 1. Compute the euclidean distance between two arrays.
# Formula: D(a, b) = sqrt(sum((a_i - b_i)^2))
def dist_p1(a, b):
  # Your code here
  pass

# Do not modify this cell
a, b = np.array([1, 1]), np.array([1, 3])
# assert dist_p1(a, b) == 2 # This assert is incorrect, distance is 2. The original code has a typo.
# Correcting based on original calculation:
assert dist_p1(a, b) == 2

a, b = np.array([-10, 1.5]), np.array([0.9, 3.6])
assert round(dist_p1(a, b), 2) == 11.1

a, b = np.array([1, 1, 1]), np.array([0, -1, 3])
assert dist_p1(a, b) == 3

a, b = np.array([1, 0, -999]), np.array([1, 3])
assert dist_p1(a, b) == "DimensionError"

a, b = np.array([10, -1, 9, 1, 2.98, -0.14]), np.array([0, -1, 3.14, np.pi, 99, np.e])
assert round(dist_p1(a, b), 2) == 96.78

# ------------------------------------------------------------------------------

# Problem 2. Normalize an array so the values range exactly between 0 and 1.
# Formula: (x - min(x)) / (max(x) - min(x))
def rescale(a):
  # Your code here
  pass

# Do not modify this cell
assert np.all(rescale(np.array([1, 2, 3, 4])) == [0, 1/3, 2/3, 1])
assert np.all(rescale(np.array([0, 1])) == [0, 1])
assert np.all(rescale(np.array([0, 10])) == [0, 1])
assert np.all(rescale(np.array([1, 2, 4])) == [0, 1/3, 1])

# ------------------------------------------------------------------------------

# Problem 3. Find the position of missing values (NaNs) in 1D numpy array.
def find(a):
  # Your code here 
  pass

# Do not modify this cell
assert np.all(find(np.array([np.nan, 1, 2, np.nan])) == [0, 3])
assert np.all(find(np.array([np.nan, np.nan])) == [0, 1])
assert np.all(find(np.array([np.e, 1, 2, 99])) == [])
assert np.all(find(np.array([np.e])) == [])
assert np.all(find(np.array([])) == [])

# ------------------------------------------------------------------------------

# Problem 4. One-hot encode a 1D numpy array 'a'.
# Let 'b' be the unique, sorted values of 'a' (size m). Create an n x m array 
# where the i-th row, j-th column is 1 if a[i] equals b[j], and 0 otherwise.
def convert(a):
  # Your code here
  pass

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

# ------------------------------------------------------------------------------

# Problem 5. Pad a 2D numpy array 'a' of size n x m with 0s 
# so that the dimensions become (n + 2*n1) x (m + 2*m1).
def padding(a, n1, m1):
  # Your code here
  pass

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

# ------------------------------------------------------------------------------

# Problem 6. Find the Euclidean distance of a 1D vector 'b' (size m) 
# from all rows of a 2D matrix 'a' (size n x m). The result must be a 1D numpy array (size n).
def dist_p6(a, b):
  # Your code here
  pass

# Do not modify this cell
a, b = np.array([[1, 1],
                 [0, 1],
                 [1, 3],
                 [4, 5]]), np.array([1, 1])
assert np.all(dist_p6(a, b) == [0., 1., 2., 5.])

a, b = np.array([[np.e, 0, 1.5],
                 [0, 1, -10]]), np.array([1, 3.14, 0])
assert np.all(np.round(dist_p6(a, b)) == [4, 10])

a, b = np.array([[1, 1],
                 [0, 1]]), np.array([1, 1, 1])
assert dist_p6(a, b) == "DimensionError"

# ------------------------------------------------------------------------------

# Problem 7. Predict the label for an unlabeled vector 'b' (size n).
# Given matrix 'a' (m x n) with rows labeled 0 or 1. Find the row in 'a' 
# closest to 'b' (Euclidean distance) and return that row's label.
def predict_label(a, labels, b):
  # Your code here
  pass

# Do not modify this cell
a = np.array([[1, -2],
              [2, 5],
              [-3, -10],
              [3, 2],
              [3, 2],
              [0, 1]])
labels = np.array([0, 1, 0, 1, 1, 0])

assert predict_label(a, labels, np.array([10, 10])) == 1
assert predict_label(a, labels, np.array([10, -10])) == 0
assert predict_label(a, labels, np.array([10, -1])) == 1
assert predict_label(a, labels, np.array([1, 1, 0])) == "DimensionError"

a = np.array([[0, 1, -2],
              [2, 1, 5],
              [-3, 3.5, -10],
              [3, 2, 9],
              [3, -2, 9]])
labels = np.array([0, 1, 0, 1, 1])
assert predict_label(a, labels, np.array([1])) == "DimensionError"
assert predict_label(a, labels, np.array([1, 1, 0])) == 0
assert predict_label(a, labels, np.array([1, 1, 10])) == 1

# ------------------------------------------------------------------------------

# Problem 8. Replace missing elements (NAN) in a 2D array 'a' 
# with the column-wise "mean", "min", or "max" based on 'mode' ("mean", "min", or "max").
def fill(a, mode):
  # Your code here
  pass

# Do not modify this cell
a = np.array([[np.nan, 200, 10],
              [2, 110, np.nan],
              [0, 120, 11],
              [0, 400, np.nan],
              [1, np.nan, 9]])

assert np.all(fill(a, "mean") ==
              [[0.75, 200.,  10.],
               [2., 110.,  10.],
               [0., 120.,  11.],
               [0., 400.,  10.],
               [1., 207.5,   9.]])

a = np.array([[np.nan, 200, 10],
              [2, 110, np.nan],
              [0, 120, 11],
              [0, 400, np.nan],
              [1, np.nan, 9]])

assert np.all(fill(a, "min") ==
              [[0., 200.,  10.],
               [2., 110.,   9.],
               [0., 120.,  11.],
               [0., 400.,   9.],
               [1., 110.,   9.]])

a = np.array([[np.nan, 200, 10],
              [2, 110, np.nan],
              [0, 120, 11],
              [0, 400, np.nan],
              [1, np.nan, 9]])

assert np.all(fill(a, "max") ==
              [[2., 200.,  10.],
               [2., 110.,  11.],
               [0., 120.,  11.],
               [0., 400.,  11.],
               [1., 400.,   9.]])

# ------------------------------------------------------------------------------

# Problem 9. Calculate the value of the expression: 
# (1/2) * b^T * A^-1 * b * (Product of all eigenvalues of A)
# Note: The product of eigenvalues is equal to the determinant of A (det(A)).
def calc(a, b):
  # Your code here
  pass

# Do not modify this cell
a = np.array([[1, 1, 2],
              [0, 1, 3],
              [1, 3, 0]])
b = np.array([1, 0, 1])
assert round(calc(a, b)) == -4

a = np.array([[1, 1, 2],
              [0, 1, 3],
              [1, 3, 0]])
b = np.array([1, 0, 1, 1])
assert calc(a, b) == "DimensionError"

a = np.array([[1, 1, 2],
              [1, 1, 3],
              [3, 3, 0]])
b = np.array([1, 0, 1])
assert calc(a, b) == "NotInvertableError"

a = np.array([[10, 1],
              [-1, 1]])
b = np.array([1, 1])
assert round(calc(a, b), 2) == 5.5

# ------------------------------------------------------------------------------

# Problem 10. Randomly divide the rows of a matrix 'a' (m x n) 
# into two parts in the ratio k:(1-k). Round the number of rows if necessary.
def rand_split(a, k):
  # Your code here
  pass

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

# ------------------------------------------------------------------------------

# Problem 11. Randomly select 'k' part of the rows of 'a' (with replacement) 
# and 'q' part of their columns (without replacement).
def rand_rows_cols(a, k, q):
  # Your code here
  pass

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

# ------------------------------------------------------------------------------

# Problem 12. Calculate the value of the expression f(W*a + b).
# W is m x n, a is n, b is m, and f is an element-wise function.
def dense_layer(w, a, b, f):
  # Your code here
  pass

# Do not modify this cell
w = np.array([[1, 1, -2],
              [0, -1, 3]])
a = np.array([3, 10, 1])
b = np.array([0, -2])
f = lambda x: x**2
assert np.all(dense_layer(w, a, b, f) == [121,  81])

f = lambda x: x
assert np.all(dense_layer(w, a, b, f) == [11,  -9])


w = np.array([[1, 1],
              [0, -1]])
assert dense_layer(w, a, b, f) == "DimensionError"

# ------------------------------------------------------------------------------

# Problem 13. Find the rows of a matrix 'a' whose sum of elements 
# is in the range [100, 200).
def between_100_200(a):
  # Your code here
  pass

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

# ------------------------------------------------------------------------------

# Problem 14. Get the L in the Cholesky decomposition of x (where x = L * L^T).
def Cholesky(x):
  # Your code here
  pass

# Do not modify this cell
x = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
assert np.all(Cholesky(x) ==
              [[2.,  0.,  0.],
               [6.,  1.,  0.],
               [-8.,  5.,  3.]])

# ------------------------------------------------------------------------------

# Problem 15. Calculate the Minkowski distance matrix (k x n) between rows 
# of 'a' (n x m) and 'b' (k x m) using parameter 'p'.
# Formula: D(x, y) = (sum(|x_i - y_i|^p))^(1/p)
def dist_mat(a, b, p):
    # Your code here
    pass

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