import numpy as np

# Problem 1. Compute the euclidean distance between two arrays.
# Formula: D(a, b) = sqrt(sum((a_i - b_i)^2))
def dist_p1(a, b):

    if a.shape != b.shape or a.ndim != 1 or b.ndim != 1:
        return "DimensionError"

    result = np.sum((a - b) ** 2)
    return np.sqrt(result)

# Do not modify this cell
a, b = np.array([1, 1]), np.array([1, 3])
assert dist_p1(a, b) == 2

a, b = np.array([-10, 1.5]), np.array([0.9, 3.6])
assert round(dist_p1(a, b), 2) == 11.1

a, b = np.array([1, 1, 1]), np.array([0, -1, 3])
assert dist_p1(a, b) == 3

a, b = np.array([1, 0, -999]), np.array([1, 3])
assert dist_p1(a, b) == "DimensionError"

a, b = np.array([10, -1, 9, 1, 2.98, -0.14]), np.array([0, -1, 3.14, np.pi, 99, np.e])
assert round(dist_p1(a, b), 2) == 96.78