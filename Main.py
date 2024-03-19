#MAIN

import Part_1_Q_1 

matrixlibrary = Part_1_Q_1.matrix(((1, 2, 4, 5), (-6, 4, 2, 3)))
print(matrixlibrary.MM(((1, 2), (3, 4), (5, 6), (7, 8))))
print(matrixlibrary.MV(((1), (2), (3), (4))))
print(matrixlibrary.ADD(((-6, 4, 2, 3), (1, 2, 4, 5))))
print(matrixlibrary.SUB(((-6, 4, 2, 3), (1, 2, 4, 5))))
denselibrary = Part_1_Q_1.DenseMatrix(((2, 4, 8), (5, 1, 0), (-8, 11, 2)))
sparselibrary = Part_1_Q_1.SparseMatrix(((0, 2, 0, 4), (2, 0, 0, 6), (3, 5, 0, 0)))

import Part_1_Q_2

matrixx = Part_1_Q_2.noLibraryMatrix(((1, -4, -6, 8), (5, 6, -10, 2)))
print(matrixx.MM(((1, 2), (3, 4), (5, 6), (7, 8))))
print(matrixx.MV(((1), (2), (3), (4))))
print(matrixx.ADD(((-6, 4, 2, 3), (1, 2, 4, 5))))
print(matrixx.SUB(((-6, 4, 2, 3), (1, 2, 4, 5))))
print(matrixx.normL1())
print(matrixx.normL2())
print(matrixx.normLinf())
densematrixx = Part_1_Q_2.DenseMatrix(((2, 5), (3, 6)))
sparsematrixx = Part_1_Q_2.SparseMatrix(((1, 2, 4), (-6, 2, 3)))
print(sparsematrixx.sparseMM(((0, 2), (0, 5), (1, 0))))
