import numpy as np
from scipy.sparse import csr_matrix


class matrix: #create a class
    def __init__(self, matrix1):
        #store the matrix1 as an attribute of the class
        self.matrix1 = np.array(matrix1)

#create methods for matrix operations  

    def MM(self, matrix2):
        #convert the matrix2 to a NumPy array
        self.matrix2 = np.array(matrix2)
        #perform matrix multiplication using np.dot
        multiplication = np.dot(self.matrix1, self.matrix2)
        #return the result of the multiplication
        return multiplication
        
    def MV(self, vector):
        #convert the vector to a NumPy array
        vector = np.array(vector)
        #check if the vector is a 1D array and has the correct dimension, vector.ndim != 1:checks if the vector's dimension is not equal to 1, vector.shape[0] != self.matrix1.shape[1]:checks if the length of the vector is not equal to the number of columns
        if vector.ndim != 1 or vector.shape[0] != self.matrix1.shape[1]:
            #raise an error if the vector's dimension is incorrect
            raise ValueError("Vector's dimension is incorrect")
        #perform matrix-vector multiplication using np.dot
        multiplication = np.dot(self.matrix1, vector)
        #return the result of the multiplication
        return multiplication
      
    def ADD(self, matrix3):
        self.matrix3 = np.array(matrix3)  #convert matrix3 to NumPy array
        addition = np.add(self.matrix1, self.matrix3)  #addition of 2 matrices
        return addition
    
    def SUB(self, matrix4):
        self.matrix4 = np.array(matrix4)  #convert matrix4 to NumPy array
        subtraction = np.subtract(self.matrix1, self.matrix4)  #subtraction of 2 matrices
        return subtraction

    def normL1(self):
        #The L1 norm of a matrix is defined as the maximum absolute column sum of the matrix.
        #In other words, it is the maximum absolute sum of the elements in each column of the matrix
        #Initialize the maximum absolute column sum to 0
        max_col_sum = 0

        # Iterate over each column
        for j in range(len(self.M1[0])):
            # Initialize the column sum for the current column
            col_sum = 0
            # Iterate over each row in the current column and sum up the absolute values
            for i in range(len(self.M1)):
                col_sum += abs(self.M1[i][j])
            # Update the maximum absolute column sum if necessary
            max_col_sum = max(max_col_sum, col_sum)

        return max_col_sum

class DenseMatrix(Matrix):  #define a subclass DenseMatrix of the Matrix class
    def __init__(self, matrix1):
        Matrix.__init__(self, matrix1) 

class SparseMatrix(Matrix):  #define a subclass SparseMatrix of the Matrix class
    def __init__(self, matrix1):
        Matrix.__init__(self, matrix1)  
        
    


