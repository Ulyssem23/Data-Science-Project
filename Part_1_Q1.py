import numpy as np
from scipy.sparse import csr_matrix


class matrix: #create a class
    def __init__(self, matrix1):
        self.matrix1 = np.array(matrix1) #store the matrix1 as an attribute of the class

#create methods for matrix operations  

    def MM(self, matrix2):
        self.matrix2 = np.array(matrix2) #convert the matrix2 to a NumPy array
        multiplication = np.dot(self.matrix1, self.matrix2)  #matrix multiplication using np.dot
        return multiplication
        
    def MV(self, vector):
        vector = np.array(vector)  #convert the vector to a NumPy array
        if vector.ndim != 1 or vector.shape[0] != self.matrix1.shape[1]:   #check if the vector is a 1D array and has the correct dimension, vector.ndim != 1:checks if the vector's dimension is not equal to 1, vector.shape[0] != self.matrix1.shape[1]:checks if the length of the vector is not equal to the number of columns
            raise ValueError("Vector's dimension is incorrect")  #raise an error if the vector's dimension is incorrect
        multiplication = np.dot(self.matrix1, vector)   #matrix-vector multiplication using np.dot
        return multiplication
      
    def ADD(self, matrix3):
        self.matrix3 = np.array(matrix3)  #convert matrix3 to NumPy array
        addition = np.add(self.matrix1, self.matrix3)  #addition of 2 matrices
        return addition
    
    def SUB(self, matrix4):
        self.matrix4 = np.array(matrix4)  #convert matrix4 to NumPy array
        subtraction = np.subtract(self.matrix1, self.matrix4)  #subtraction of 2 matrices
        return subtraction

class DenseMatrix(matrix):  #define a subclass DenseMatrix of the Matrix class
    def __init__(self, matrix1):
        matrix.__init__(self, matrix1) 

class SparseMatrix(matrix):  #define a subclass SparseMatrix of the Matrix class
    def __init__(self, matrix1):
        matrix.__init__(self, matrix1)
"""    def sparse(self):
        sparseM = csr_matrix(self.matrix1)
        return sparseM"""
