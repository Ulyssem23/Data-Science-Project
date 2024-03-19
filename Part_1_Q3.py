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


def power_iteration(A, num_iterations=1000):
    n = len(A)
    b_k = [1.0] * n  # Initial vector
    for _ in range(num_iterations):
        # Calculate the matrix-by-vector product Ab
        b_k1 = [sum(A[i][j] * b_k[j] for j in range(n)) for i in range(n)]
        # Calculate the norm of b_k1
        norm_b_k1 = sum(x**2 for x in b_k1) ** 0.5
        # Normalize the vector
        b_k = [x / norm_b_k1 for x in b_k1]
    # Return the approximate eigenvalue
    return sum(A[i][j] * b_k[j] for i in range(n) for j in range(n)) / sum(b_k[i]**2 for i in range(n))


A = [
    [4, 1],
    [2, 3]
]


