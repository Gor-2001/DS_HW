import numpy as np
# Problem 12. Calculate the value of the expression f(W*a + b).
# W is m x n, a is n x 1, b is m x 1, and f is an element-wise function.
def dense_layer(w, a, b, f):

    if w.ndim != 2 or a.ndim != 1 or b.ndim != 1:
        return "DimensionError"
    
    if w.shape[1] != a.shape[0]:
        return "DimensionError"
    
    if w.shape[0] != b.shape[0]:
        return "DimensionError"
    
    c = w @ a + b
    return f(c)

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