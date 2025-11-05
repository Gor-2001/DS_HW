# Practical Exercise: Student Grade Analysis System
#
# You are a data analyst at a university and need to analyze student performance across different subjects. This exercise will test your understanding of NumPy array operations, aggregations, and data manipulation.
#
# Scenario: You have grade data for 50 students across 5 subjects (Math, Physics, Chemistry, Biology, English). Some grades are missing (represented as NaN).

import numpy as np
import timeit # Added for Task 6, as it was used implicitly by %timeit
import time

# ----------------------------------------------------------------------
# Initialization
np.random.seed(int(time.time()))  # For reproducibility
print("\n**************************************")
print("Task 1:\n**************************************\n")

# ----------------------------------------------------------------------
# ## Task 1: Data Generation and Setup (2 points)

# 1. Generate a 50x5 array of random grades between 0-100 (integers)
grades = np.random.randint(0, 101, size=(50, 5)).astype(float)

# 2. Randomly replace 5% of the grades with NaN to simulate missing data
total_elements = grades.size
num_nan = int(total_elements * 0.05)
nan_indices = np.random.choice(total_elements, num_nan, replace=False)
grades.flat[nan_indices] = np.nan

# 3. Create arrays for student IDs (1001–1050) and subject names
student_ids = np.arange(1001, 1051)
subjects = np.array(['Math', 'Physics', 'Chemistry', 'Biology', 'English'])

# 4. Print the shape and basic info about the dataset
print("Grades:\n", grades, "\n")
print("Shape:", grades.shape)
print("Data type:", grades.dtype, "\n")

# ----------------------------------------------------------------------
means = np.nanmean(grades, axis=0)
mins = np.nanmin(grades, axis=0)
maxs = np.nanmax(grades, axis=0)
medians = np.nanmedian(grades, axis=0)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
print(f"{'Subject':<12}", end="")
for subj in subjects:
    print(f"{subj:<12}", end="")
print()

def print_row(label, arr):
    print(f"{label:<12}", end="")
    for x in arr:
        print(f"{x:<12.1f}", end="")
    print()

print_row("Mean", means)
print_row("Min", mins)
print_row("Max", maxs)
print_row("Median", medians)

# ----------------------------------------------------------------------
# ## Task 2: Basic Statistics and Aggregations (3 points)
print("\n**************************************")
print("Task 2:\n**************************************\n")

overall_mean = np.nanmean(grades)
mean_per_subject = means
mean_per_student = np.nanmean(grades, axis=1)
overall_min = np.nanmin(grades)
overall_max = np.nanmax(grades)
missing_count = np.isnan(grades).sum()

print(f"Overall mean: {overall_mean:.2f}")
print(f"Mean per subject:\n{mean_per_subject}")
print(f"Mean per student:\n{mean_per_student}\n")
print(f"Min grade: {overall_min:.1f}")
print(f"Max grade: {overall_max:.1f}")
print(f"Missing grades count: {missing_count}")

# ----------------------------------------------------------------------
# ## Task 3: Grade Classification and Filtering (3 points)
print("\n**************************************")
print("Task 3:\n**************************************\n")

conditions = [
    (grades >= 90),
    (grades >= 80) & (grades < 90),
    (grades >= 70) & (grades < 80),
    (grades >= 60) & (grades < 70),
    (grades >= 0) & (grades < 60),
    np.isnan(grades)
]
choices = ['A', 'B', 'C', 'D', 'E', 'N']

grades_classified = np.select(conditions, choices, default='INVALID')
print("Grade classification:\n", grades_classified, "\n")

print("Count grades in each category")
for it in range(0, len(choices)):
    print(choices[it], np.sum(conditions[it]))
print("\n")

# Students with average > 85
student_avg = mean_per_student
good_students = student_avg > 85
bad_students = student_avg < 70

print("Students with average > 85:")
print("Averages:", student_avg[good_students])
print("Indexes:", np.where(good_students)[0], "\n")

print("Students with average < 70:")
print("Averages:", student_avg[bad_students])
print("Indexes:", np.where(bad_students)[0], "\n")

# Students who failed any subject (E or N)
failed_any = np.any((grades_classified == 'E') | (grades_classified == 'N'), axis=1)
print("Indexes of students who failed any subject:")
print(np.where(failed_any)[0], "\n")

# ----------------------------------------------------------------------
# ## Task 4: Data Manipulation and Reshaping (2 points)
print("\n**************************************")
print("Task 4:\n**************************************\n")

# 1. Normalize grades per subject (0–1 scale)
range_per_subject = maxs - mins
normalized_grades = (grades - mins) / range_per_subject
print("Normalized grades (per subject):\n", normalized_grades, "\n")

# 2. Top 5 students by average grade
sorted_avg = np.sort(student_avg)
top5 = sorted_avg[-5:]
print("Top 5 student averages:\n", top5, "\n")

# 3. 90th percentile of all grades
percentile_90 = np.nanpercentile(grades, 90)
print("90th percentile of all grades:", percentile_90, "\n")

# 4. Difference from subject average
diff_from_mean = grades - means
print("Difference from subject average:\n", diff_from_mean)

# ----------------------------------------------------------------------
# ## Task 5: Advanced Analysis (3 points)
#
# 1. **Grade Improvement Simulation**: Add 5 points to all grades below 70 and ensure no grade exceeds 100
# 2. **Subject Correlation**: Calculate the correlation between Math and Physics grades (hint: use np.corrcoef)
# 3. **Performance Matrix**: Create a 5x5 matrix showing the average grade difference between each pair of subjects
# 4. **Missing Data Imputation**: Replace all NaN values with the subject's median grade
# 5. **Performance Ranking**: Rank students from best to worst based on their average grades

# Your code here
# 1. Grade improvement simulation

# 2. Math-Physics correlation

# 3. Subject comparison matrix

# 4. Impute missing values with median

# 5. Student ranking

# ----------------------------------------------------------------------
# ## Task 6: Performance Optimization Challenge (2 points)
print("\n**************************************")
print("Task 6:\n**************************************\n")

# 1. Pure Python function
def calculate_averages_python(grades_matrix):
    total = 0.0
    count = 0
    for row in grades_matrix:
        for el in row:
            if el is not None and not (isinstance(el, float) and np.isnan(el)):
                total += el
                count += 1
    return total / count if count > 0 else float('nan')


# 2. NumPy function
def calculate_averages_numpy(grades_matrix):
    return np.nanmean(grades_matrix)


# 3. Prepare test data
grades = np.random.randint(0, 101, size=(100, 100)).astype(float)
grades.ravel()[np.random.choice(grades.size, 5000, replace=False)] = np.nan
python_data = grades.tolist()

# 4. Performance comparison
t_python = timeit.timeit(lambda: calculate_averages_python(python_data), number=3)
t_numpy = timeit.timeit(lambda: calculate_averages_numpy(grades), number=3)

print("Pure Python average:\n", calculate_averages_python(python_data), "\n")
print("NumPy average:\n", calculate_averages_numpy(grades), "\n")
print("Pure Python time:\n", f"{t_python:.4f} sec\n")
print("NumPy time:\n", f"{t_numpy:.4f} sec\n")
print("Speedup:\n", f"{t_python / t_numpy:.1f}x faster with NumPy\n")


# ----------------------------------------------------------------------
# ## Task 7: Write a function to return a 2D array of given size 
# ## with 1s on the border and 0s inside. (2 points)
print("\n**************************************")
print("Task 7:\n**************************************\n")

# 1. Function definition
def zero_one(x):
    result = np.zeros((x, x), dtype=int)
    result[0, :] = 1
    result[-1, :] = 1
    result[:, 0] = 1
    result[:, -1] = 1
    return result

# 2. Display result
print("Generated matrix with 1s on the border and 0s inside:\n")
print(zero_one(10))
print("")

# ----------------------------------------------------------------------
# ## Task 8: Given 2D numpy arrays 'a' and 'b' of sizes $m \times n$ and $k \times k$ respectively (k <= n, k <= m), 2 integers 'stride' and 'padding' and 'f' function. (3 points)
# You need to
#   1. first pad 'a' matrix with 0s on each side,
#   2. then move 'b'  over 'a' with stride 'stride', then multiply their elements by the corresponding 'b' elements,
#   3. add the resulting k * k numbers
#   4. apply the 'f' function to the result
#   5. and place them in the new matrix.
#  ```
# a = np.array([[1, 1, 2],
#               [0, 1, 3],
#               [1, 3, 0],
#               [4, 5, 2]])
# b = np.array([[1, 0],
#               [0, 1]])
#
# stride = 1
# padding = 0
# f = lambda x: x**2
# print(conv(a, b, stride, padding, f))
# >>[[4, 16],
#   [9, 1],
#   [36, 25]]
# ```
# ![ConvUrl](https://miro.medium.com/max/2340/1*Fw-ehcNBR9byHtho-Rxbtw.gif "conv")

print("\n**************************************")
print("Task 8:\n**************************************\n")

def conv(a, b, stride=1, padding=0, f=lambda x: x):

    a_padded = np.zeros((a.shape[0] + 2*padding, a.shape[1] + 2*padding), dtype=a.dtype)
    a_padded[padding:padding+a.shape[0], padding:padding+a.shape[1]] = a

    return a_padded


# # Tests for the TASK 8
# # Do not modify this cell
# a = np.array([[1, 1, 2],
#               [0, 1, 3],
#               [1, 3, 0]])
# b = np.array([[1, 0],
#               [0, 1]])
# f = lambda x: x**2
# # The tests are kept as is, but cannot be run without the implementation
# assert np.all(conv(a, b, 1, 0, f) ==
#               [[4., 16.],
#                [9.,  1.]])

# a = np.array([[1, 1, 2],
#               [0, 1, 3],
#               [1, 3, 0],
#               [-10, -3, 0]])
# b = np.arange(9).reshape(3, 3)
# f = lambda x:0 if x<0 else x
# assert np.all(conv(a, b, 1, 0, f) ==
#               [[51.],
#                [0.]])

# b = np.arange(6).reshape(2, 3)
# #assert conv(a, b, 1, 0, f) == "DimensionError"
