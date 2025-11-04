import numpy as np
# Problem 8. Replace missing elements (NAN) in a 2D array 'a' 
# with the column-wise "mean", "min", or "max" based on 'mode' ("mean", "min", or "max").
def fill(a, mode):
    mode_funcs = {
        "mean": np.nanmean,
        "min": np.nanmin,
        "max": np.nanmax
    }
    
    func = mode_funcs[mode]
    a = np.nan_to_num(a, nan=func(a, axis=0))
    return a

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