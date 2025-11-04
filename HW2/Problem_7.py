import numpy as np
# Problem 7. Predict the label for an unlabeled vector 'b' (size n).
# Given matrix 'a' (m x n) with rows labeled 0 or 1. Find the row in 'a' 
# closest to 'b' (Euclidean distance) and return that row's label.
def predict_label(a, labels, b):
    
    if a.ndim != 2 or b.ndim != 1 or a.shape[1] != b.shape[0]:
      return "DimensionError"
    
    dist = np.sqrt(np.sum((a - b)**2, axis=1))
    index = np.argmin(dist)
    return labels[index]


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