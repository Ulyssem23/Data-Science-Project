#MAIN

import numpy as np

Random_500x500_matrix = np.random.randint(0, 101, size=(500, 500))

Random_500x1_vector = np.random.randint(0, 101, size=(500, ))

Random_500x500_matrix2 = np.random.randint(0, 101, size=(500, 500))


import Part_1_Q_1 

matrixlibrary = Part_1_Q_1.matrix(Random_500x500_matrix)
print(matrixlibrary.MM(Random_500x500_matrix2))
print(matrixlibrary.MV(Random_500x1_vector))
print(matrixlibrary.ADD(Random_500x500_matrix2))
print(matrixlibrary.SUB(Random_500x500_matrix2))
#denselibrary = Part_1_Q_1.DenseMatrix((Random_500x500_matrix))
#sparselibrary = Part_1_Q_1.SparseMatrix((Random_500x500_matrix))

import Part_1_Q_2

matrixx = Part_1_Q_2.noLibraryMatrix((Random_500x500_matrix))

print(matrixx.MM(Random_500x500_matrix2))
print(matrixx.MV(Random_500x1_vector))
print(matrixx.ADD(Random_500x500_matrix2))
print(matrixx.SUB(Random_500x500_matrix2))
print(matrixx.normL1())
print(matrixx.normL2())
print(matrixx.normLinf())
print(matrixx.power_iteration())

#densematrixx = Part_1_Q_2.DenseMatrix(((2, 5), (3, 6)))
#sparsematrixx = Part_1_Q_2.SparseMatrix(((1, 2, 4), (-6, 2, 3)))
#print(sparsematrixx.sparseMM(((0, 2), (0, 5), (1, 0))))
#print(sparsematrixx.jacobi_method(vector, matrix))

import Part_1_Q_3
mattrix = Part_1_Q_3.matrix(((1, 0), (0,1)))

print(mattrix.eigenvalues())

import Part_1_Q_2

matrixx = Part_1_Q_2.noLibraryMatrix(((1, 0), (0, 1)))

print(matrixx.compute_svd(num_iterations=1000))

vector = [1, 2]
matrix = {(7, 9): 1, (9, 1): 2, (1, 0): -6, (1, 1): 2, (1, 2): 3}

