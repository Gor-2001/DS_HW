import numpy as np

# Problem 2. Normalize an array so the values range exactly between 0 and 1.
# Formula: (x - min(x)) / (max(x) - min(x))
def rescale(a):
    a = a - np.min(a)
    factor = np.max(a) - np.min(a)
    a = a / factor
    return a

# Do not modify this cell
assert np.all(rescale(np.array([1, 2, 3, 4])) == [0, 1/3, 2/3, 1])
assert np.all(rescale(np.array([0, 1])) == [0, 1])
assert np.all(rescale(np.array([0, 10])) == [0, 1])
assert np.all(rescale(np.array([1, 2, 4])) == [0, 1/3, 1])