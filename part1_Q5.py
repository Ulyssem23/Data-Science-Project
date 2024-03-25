#Part_1_Q_5
import Part_1
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from scipy.sparse import random

Random_100x100_matrix = np.random.randint(0, 101, size=(100, 100))
Random_100x100_matrix2 = np.random.randint(0, 101, size=(100, 100))
Random_200x200_matrix = np.random.randint(0, 101, size=(200, 200))
Random_200x200_matrix2 = np.random.randint(0, 101, size=(200, 200))
Random_300x300_matrix = np.random.randint(0, 101, size=(300, 300))
Random_300x300_matrix2 = np.random.randint(0, 101, size=(300, 300))
Random_400x400_matrix = np.random.randint(0, 101, size=(400, 400))
Random_400x400_matrix2 = np.random.randint(0, 101, size=(400, 400))
Random_500x500_matrix = np.random.randint(0, 101, size=(500, 500))
Random_500x500_matrix2 = np.random.randint(0, 101, size=(500, 500))


Random_100x100_Sparse_matrix = random(100, 100, density=0.01, format='csr')
Random_200x200_Sparse_matrix = random(200, 200, density=0.01, format='csr')
Random_300x300_Sparse_matrix = random(300, 300, density=0.01, format='csr')
Random_400x400_Sparse_matrix = random(400, 400, density=0.01, format='csr')
Random_500x500_Sparse_matrix = random(500, 500, density=0.01, format='csr')

Random_100x1_vector = np.random.randint(0, 101, size=(100, ))
Random_200x1_vector = np.random.randint(0, 101, size=(200, ))
Random_300x1_vector = np.random.randint(0, 101, size=(300, ))
Random_400x1_vector = np.random.randint(0, 101, size=(400, ))
Random_500x1_vector = np.random.randint(0, 101, size=(500, ))

Random_100x0_vector = [np.random.randint(0, 100) for _ in range(100)]
Random_200x0_vector = [np.random.randint(0, 100) for _ in range(200)]
Random_300x0_vector = [np.random.randint(0, 100) for _ in range(300)]
Random_400x0_vector = [np.random.randint(0, 100) for _ in range(400)]
Random_500x0_vector = [np.random.randint(0, 100) for _ in range(500)]

P1100 = Part_1.matrix_Q1(Random_100x100_matrix)
P1200 = Part_1.matrix_Q1(Random_200x200_matrix)
P1300 = Part_1.matrix_Q1(Random_300x300_matrix)
P1400 = Part_1.matrix_Q1(Random_400x400_matrix)
P1500 = Part_1.matrix_Q1(Random_500x500_matrix)


def MM_Q1(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1{i+1}00'], 'MM')(globals()[f'Random_{size}x{size}_matrix2'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q1 MM')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#MM_Q1(10)

def MV_Q1(n):
    vector_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(vector_sizes))]

    for i, size in enumerate(vector_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1{i+1}00'], 'MV')(globals()[f'Random_{size}x1_vector'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x1')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q1 MV')
    plt.legend(title='Vector Size')
    plt.grid(True)
    plt.show()
#MV_Q1(10)

def ADD_Q1(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1{i+1}00'], 'ADD')(globals()[f'Random_{size}x{size}_matrix2'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q1 ADD')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#ADD_Q1(10)

def SUB_Q1(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1{i+1}00'], 'SUB')(globals()[f'Random_{size}x{size}_matrix2'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q1 SUB')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#SUB_Q1(10)

P1Q2100 = Part_1.noLibraryMatrix((Random_100x100_matrix))
P1Q2200 = Part_1.noLibraryMatrix((Random_200x200_matrix))
P1Q2300 = Part_1.noLibraryMatrix((Random_300x300_matrix))
P1Q2400 = Part_1.noLibraryMatrix((Random_400x400_matrix))
P1Q2500 = Part_1.noLibraryMatrix((Random_500x500_matrix))

def MM_Q2(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'MM')(globals()[f'Random_{size}x{size}_matrix2'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 MM')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#MM_Q2(2)


def MV_Q2(n):
    vector_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(vector_sizes))]

    for i, size in enumerate(vector_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'MV')(globals()[f'Random_{size}x0_vector'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x0')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 MV')
    plt.legend(title='Vector Size')
    plt.grid(True)
    plt.show()
#MV_Q2(10)


def ADD_Q2(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'ADD')(globals()[f'Random_{size}x{size}_matrix2'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 ADD')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#ADD_Q2(10)

def SUB_Q2(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'SUB')(globals()[f'Random_{size}x{size}_matrix2'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 SUB')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#SUB_Q2(10)


def NORML1_Q2(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'normL1')()
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 normL1')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#NORML1_Q2(10)


def NORML2_Q2(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'normL2')()
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 normL2')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#NORML2_Q2(10)

def NORMLINF_Q2(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'normLinf')()
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 normLINF')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#NORMLINF_Q2(10)


def TRANSPOSE_Q2(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'transpose')()
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 Transpose')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#TRANSPOSE_Q2(10)


def power_iteration_Q2(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'power_iteration')()
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 Power Iteration')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#power_iteration_Q2(2)


def compute_svd_Q2(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1Q2{i+1}00'], 'compute_svd')()
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q2 Compute SVD')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#compute_svd_Q2(2)

P1D100 = Part_1.DenseMatrix((Random_100x100_matrix))
P1D200 = Part_1.DenseMatrix((Random_200x200_matrix))
P1D300 = Part_1.DenseMatrix((Random_300x300_matrix))
P1D400 = Part_1.DenseMatrix((Random_400x400_matrix))
P1D500 = Part_1.DenseMatrix((Random_500x500_matrix))

def gaussian_elimination_Q3(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1D{i+1}00'], 'gaussian_elimination')(globals()[f'Random_{size}x1_vector'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'500x1')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q3 Gaussian Elimination')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#gaussian_elimination_Q3(5)

P1S100 = Part_1.SparseMatrix((Random_100x100_matrix))
P1S200 = Part_1.SparseMatrix((Random_200x200_matrix))
P1S300 = Part_1.SparseMatrix((Random_300x300_matrix))
P1S400 = Part_1.SparseMatrix((Random_400x400_matrix))
P1S500 = Part_1.SparseMatrix((Random_500x500_matrix))

def sparse_MM_Q3(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1S{i+1}00'], 'sparseMM')(globals()[f'Random_{size}x{size}_matrix2'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'{size}x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q3 Sparse MM')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#sparse_MM_Q3(2)


def jacobi_method_Q3(n):
    matrix_sizes = [100, 200, 300, 400, 500]
    times = [[] for _ in range(len(matrix_sizes))]

    for i, size in enumerate(matrix_sizes):
        for _ in range(n):
            start_time = time.time()
            getattr(globals()[f'P1S{i+1}00'], 'jacobi_method')(globals()[f'Random_{size}x1_vector'], globals()[f'Random_{size}x{size}_matrix'])
            end_time = time.time()
            duration = end_time - start_time
            times[i].append(duration)

        plt.plot(range(1, n+1), times[i], marker='o', label=f'500x{size}')

    plt.xlabel('Iteration')
    plt.ylabel('Time (seconds)')
    plt.title('P1Q3 Jacobi Method')
    plt.legend(title='Matrix Size')
    plt.grid(True)
    plt.show()
#jacobi_method_Q3(10)
