class DenseMatrixSolver:
    def __init__(self, matrix, vector):
        self.matrix = matrix
        self.vector = vector

    def gaussian_elimination(self):
        n = len(self.matrix)
        for i in range(n):
            # Partial pivoting
            max_row = max(range(i, n), key=lambda x: abs(self.matrix[x][i]))
            self.matrix[i], self.matrix[max_row] = self.matrix[max_row], self.matrix[i]
            self.vector[i], self.vector[max_row] = self.vector[max_row], self.vector[i]
            
            # Normalize the pivot row
            for j in range(i+1, n):
                factor = self.matrix[j][i] / self.matrix[i][i]
                for k in range(i, n):
                    self.matrix[j][k] -= factor * self.matrix[i][k]
                self.vector[j] -= factor * self.vector[i]

        # Back substitution
        x = [0 for _ in range(n)]
        for i in range(n-1, -1, -1):
            x[i] = self.vector[i]
            for j in range(i+1, n):
                x[i] -= self.matrix[i][j] * x[j]
            x[i] /= self.matrix[i][i]
        return x


class SparseMatrixSolver:
    def __init__(self, matrix, vector):
        # Sparse matrix stored as a dictionary of keys (row, col) with non-zero values
        self.matrix = { (i,j): matrix[i][j] for i in range(len(matrix)) for j in range(len(matrix[0])) if matrix[i][j] != 0 }
        self.vector = vector

    def jacobi_method(self, iterations=1000, tolerance=1e-10):
        n = len(self.vector)
        x = [0 for _ in range(n)]  # Initial guess
        x_new = x.copy()

        for _ in range(iterations):
            for i in range(n):
                sum = 0
                for j in range(n):
                    if i != j:
                        sum += self.matrix.get((i, j), 0) * x[j]
                x_new[i] = (self.vector[i] - sum) / self.matrix.get((i, i), 1)
            if all(abs(x_new[i] - x[i]) < tolerance for i in range(n)):
                break
            x = x_new.copy()
        return x
