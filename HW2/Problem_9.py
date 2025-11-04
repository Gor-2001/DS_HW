import numpy as np
# Problem 9. Calculate the value of the expression: 
# (1/2) * b^T * A^-1 * b * (Product of all eigenvalues of A)
# Note: The product of eigenvalues is equal to the determinant of A (det(A)).
def calc(a, b):

    if a.ndim != 2 or b.ndim != 1 or a.shape[0] != a.shape[1]:
        return "DimensionError"

    if b.shape[0] != a.shape[0]:
        return "DimensionError"
    
    if np.linalg.det(a) == 0:
        return "NotInvertableError"
    
    c = 0.5 * b.T @ np.linalg.inv(a) @ b * np.linalg.det(a)
    return c

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