#MAIN
import Part_1
import Part_1_Q_5
import numpy as np
import random
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

matrixlibrary = Part_1.matrix_Q1(Random_500x500_matrix)
print(matrixlibrary.MM(Random_500x500_matrix2))
print(matrixlibrary.MV(Random_500x1_vector))
print(matrixlibrary.ADD(Random_500x500_matrix2))
print(matrixlibrary.SUB(Random_500x500_matrix2))
denselibrary = Part_1.DenseMatrix((Random_500x500_matrix))
sparselibrary = Part_1.SparseMatrix((Random_500x500_matrix))

matrixNOlibrary = Part_1.noLibraryMatrix((Random_500x500_matrix))

print(matrixNOlibrary.MM(Random_500x500_matrix2))
print(matrixNOlibrary.MV(Random_500x0_vector))
print(matrixNOlibrary.ADD(Random_500x500_matrix2))
print(matrixNOlibrary.SUB(Random_500x500_matrix2))
print(matrixNOlibrary.normL1())
print(matrixNOlibrary.normL2())
print(matrixNOlibrary.normLinf())
print(matrixNOlibrary.transpose())
print(matrixNOlibrary.power_iteration())
print(matrixNOlibrary.compute_svd())

denseNOlibrary = Part_1.DenseMatrix(Random_500x500_matrix2)
print(denseNOlibrary.gaussian_elimination(Random_500x1_vector))


sparseNOlibrary = Part_1.SparseMatrix(Random_500x500_matrix2)
print(sparseNOlibrary.sparseMM(((0, 2), (0, 5), (1, 0))))
print(sparseNOlibrary.jacobi_method(Random_500x0_vector, Random_500x500_matrix2))

matrixQ3 = Part_1.matrix_Q3(((1, 0), (0,1)))
print(matrixQ3.eigenvalues())
print(matrixQ3.decomposition())

Part_1_Q_5.MM_Q1(10)
Part_1_Q_5.MV_Q1(10)
Part_1_Q_5.SUB_Q1(10)
Part_1_Q_5.MM_Q2(2)
Part_1_Q_5.MV_Q2(10)
Part_1_Q_5.ADD_Q2(10)
Part_1_Q_5.SUB_Q2(10)
Part_1_Q_5.NORML1_Q2(10)
Part_1_Q_5.NORML2_Q2(10)
Part_1_Q_5.NORMLINF_Q2(10)
Part_1_Q_5.TRANSPOSE_Q2(10)
Part_1_Q_5.power_iteration_Q2(2)
Part_1_Q_5.compute_svd_Q2(2)
Part_1_Q_5.gaussian_elimination_Q3(5)
Part_1_Q_5.sparse_MM_Q3(2)
Part_1_Q_5.jacobi_method_Q3(10)
