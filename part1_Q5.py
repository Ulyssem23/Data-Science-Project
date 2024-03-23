import time
import numpy as np
import matplotlib.pyplot as plt

# Define the noLibraryMatrix class
class noLibraryMatrix:
    # Include class methods here

# Define test cases for matrices of size 500×500
def generate_test_case(size):
    return np.random.rand(size, size)

# Define functions to measure time for operations on sparse and dense matrices
def measure_time(operation, matrix):
    start_time = time.time()
    result = operation(matrix)
    end_time = time.time()
    return end_time - start_time

# Define functions to compare performance of sparse and dense matrices
def compare_performance(matrix_size):
    dense_matrix = generate_test_case(matrix_size)
    sparse_matrix = {i: {j: np.random.rand() for j in range(matrix_size)} for i in range(matrix_size)}

    dense_time = measure_time(noLibraryMatrix.MM, dense_matrix)
    sparse_time = measure_time(SparseMatrix.sparseMM, sparse_matrix)

    return dense_time, sparse_time

# Measure performance for increasing matrix sizes
matrix_sizes = [500 * 2**i for i in range(7)]  # Starting from 500 and doubling the size until reaching computational limits
dense_times = []
sparse_times = []

for size in matrix_sizes:
    dense_time, sparse_time = compare_performance(size)
    dense_times.append(dense_time)
    sparse_times.append(sparse_time)

# Plot the performance comparison results
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, dense_times, marker='o', label='Dense Matrix')
plt.plot(matrix_sizes, sparse_times, marker='o', label='Sparse Matrix')
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.title('Performance Comparison of Sparse and Dense Matrices')
plt.legend()
plt.grid(True)
plt.show()
_____________________________________________________________________________________________________________________________________________________________
#Part_1_Q_5
import Part_1_Q_1
import Part_1_Q_2
import numpy as np
import random
import time
import matplotlib.pyplot as plt



Random_500x500_matrix = np.random.randint(0, 101, size=(500, 500))
Random_500x500_matrix2 = np.random.randint(0, 101, size=(500, 500))
Random_500x1_vector = np.random.randint(0, 101, size=(500, ))
Random_500x0_vector = [random.randint(0, 100) for _ in range(500)]

P1Q1_Matrix = Part_1_Q_1.matrix(Random_500x500_matrix)

def MMP1500(n):
    times = []
    for _ in range(n):
        start_time = time.time()
        P1Q1_Matrix.MM(Random_500x500_matrix2)
        # Perform your task here
        # For example, a dummy task:
        time.sleep(1)  # Sleep for 1 second
        end_time = time.time()
        duration = end_time - start_time
        times.append(duration)
    
    plt.plot(range(1, n+1), times, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q1 MM')
    plt.grid(True)
    plt.show()
#MMP1500(10)

def MVP1500(n):
    times = []
    for _ in range(n):
        start_time = time.time()
        P1Q1_Matrix.MV(Random_500x1_vector)
        # Perform your task here
        # For example, a dummy task:
        time.sleep(1)  # Sleep for 1 second
        end_time = time.time()
        duration = end_time - start_time
        times.append(duration)
    
    plt.plot(range(1, n+1), times, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q1 MV')
    plt.grid(True)
    plt.show()
MVP1500(10)



P1Q2_Matrix = Part_1_Q_2.noLibraryMatrix((Random_500x500_matrix))
