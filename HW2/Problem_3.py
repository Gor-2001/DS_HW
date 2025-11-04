import numpy as np

# Problem 3. Find the position of missing values (NaNs) in 1D numpy array.
def find(a):
    return np.where(np.isnan(a))[0]

# Do not modify this cell
assert np.all(find(np.array([np.nan, 1, 2, np.nan])) == [0, 3])
assert np.all(find(np.array([np.nan, np.nan])) == [0, 1])
assert np.all(find(np.array([np.e, 1, 2, 99])) == [])
assert np.all(find(np.array([np.e])) == [])
assert np.all(find(np.array([])) == [])