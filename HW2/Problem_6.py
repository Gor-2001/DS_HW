import numpy as np
# Problem 6. Find the Euclidean distance of a 1D vector 'b' (size m) 
# from all rows of a 2D matrix 'a' (size n x m). The result must be a 1D numpy array (size n).
import numpy as np

def dist_p6(a, b):

    if a.ndim != 2 or b.ndim != 1 or a.shape[1] != b.shape[0]:
        return "DimensionError"

    result = np.sqrt(np.sum((a - b)**2, axis=1))
    return result

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