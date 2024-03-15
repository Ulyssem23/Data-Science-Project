import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eigvals
from scipy.linalg import svd


class matrix: #create a class
    def __init__(self, matrix1):
        #store the matrix1 as an attribute of the class
        self.matrix1 = np.array(matrix1)
        
    def eigenvalues(self):
        values = eigvals(self.matrix1)
        return values
    
    def decomposition(self):
        U, S, Vt = svd(self.matrix1)
        return U, S, Vt
