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



class noLibraryMatrix:
    def __init__(self, M1):
        self.M1 = M1

    def transpose(self):
        return [list(row) for row in zip(*self.M1)]

    def multiply_matrices(self, M2):
        result = [[sum(a*b for a, b in zip(M1_row, M2_col)) for M2_col in zip(*M2)] for M1_row in self.M1]
        return result

    @staticmethod
    def power_iteration(A, num_iterations=1000):
        n = len(A[0])
        b_k = [1.0] * n
        for _ in range(num_iterations):
            b_k1 = [sum(A_row[j] * b_k[j] for j in range(n)) for A_row in A]
            norm = sum(x**2 for x in b_k1) ** 0.5
            b_k = [x / norm for x in b_k1]
        return b_k, norm

    def compute_svd(self):
        AT = self.transpose()
        AAT = self.multiply_matrices(AT)
        ATA = noLibraryMatrix(AT).multiply_matrices(self.M1)
        
        # Compute left singular vectors (U)
        u, _ = self.power_iteration(AAT)
        U = [u]

        # Compute right singular vectors (V)
        v, sigma = self.power_iteration(ATA)
        V = [v]

        # Singular values (Sigma) - approximation
        Sigma = [sigma]

        # For a more accurate and complete SVD, iterate over all singular values/vectors
        # This example computes only the first singular value/vector pair
        
        return U, Sigma, V

# Example of using the class
matrix_data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Example matrix
matrix = noLibraryMatrix(matrix_data)
U, Sigma, V = matrix.compute_svd()

print("U (approx):", U)
print("Sigma (approx):", Sigma)
print("V^T (approx):", [list(row) for row in zip(*V)])  # Transpose V to get V^T

