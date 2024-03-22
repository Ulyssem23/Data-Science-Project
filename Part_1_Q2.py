class noLibraryMatrix: #create a new class
    def __init__(self, M1: list[list]):
        self.M1 = M1  #store M1 as an attribute of the class

#create basic operations for matrices without using libraries

    def MM(self, M2):  #matrix multiplication
        self.M2 = M2  #store M2 as an attribute of the class
        
        #initialize result matrix with zeros
#len(self.M1): length of M1 = number of rows in the matrix, len(M2[0]): length of the first row of M2 = number of columns
# 0 for _ in range(len(M2[0])): list with zeros, where the number of zeros is equal to the number of columns in M2, _ is a throwaway variable, dont need the value of the loop variable
#for _ in range(len(self.M1)): iterates over the number of rows in M1, for each row, it creates a list of zeros
        result = [[0 for _ in range(len(M2[0]))] for _ in range(len(self.M1))]
        
        for i in range(len(self.M1)):   #loop over the rows of M1
            for j in range(len(self.M2[0])):    #loop over the columns of M2
                for k in range(len(self.M2)):   #loop over the rows of the second matrix M2 + columns of the first matrix M1
                    result[i][j] += self.M1[i][k] * self.M2[k][j]   #compute dot product between the ith row of M1 and the kth column of M2 + update each element of the result matrix
        return result

    def MV(self, vector): #vector matrix multiplication
            self.vector = vector  #store the vector as an attribute of the class
            is_one_dimensional = all(isinstance(x, (int, float)) for x in vector)   #check if the vector is one-dimensional 
            if not is_one_dimensional:
                raise ValueError("Vector's dimension is incorrect")   #if the vector is not one-dimensional, raise a ValueError
            result = [0] * len(self.M1)   #initialize a result list with zeros
        
            for i in range(len(self.M1)):  #iterate over the rows of M1
                for k in range(len(vector)):   #iterate over the elements of the vector
                    result[i] += self.M1[i][k] * vector[k]  #multiply  elements of M1 and the vector and add to the result
            return result

    def ADD(self, M3): #addition
        self.M3 = M3  #store  M3 as an attribute of the class
        result = []  #initialize result list
        
        for i in range(len(self.M1)):   #iterate over each row of the matrices M1 and M3
            row = []  #initialize a row for the result matrix
            for j in range(len(self.M1[0])):    #iterate over each column of the matrices M1 and M3
                row.append(self.M1[i][j] + M3[i][j])  #add corresponding elements of M1 and M3 and append to the row
            result.append(row)   #append row to the result matrix
        return result 

    def SUB(self, M3): #substraction
        self.M3 = M3   #store  M3 as an attribute of the class
        result = []   #initialize an empty list to store the result matrix
        for i in range(len(self.M1)):   #iterate over the rows of M1
            row = []  #initialize an empty list to store each row of the result matrix
            for j in range(len(self.M1[0])):  #iterate over the columns of M1
                row.append(self.M1[i][j] - M3[i][j])   #subtract elements of M1 and M3 and append to the row
            result.append(row)
        return result

    def normL1(self):
        #The L1 norm of a matrix is defined as the maximum absolute column sum of the matrix.
       
        max_col_sum = 0  #initialize the maximum absolute column sum to 0

        for j in range(len(self.M1[0])): #iterate over each column
            col_sum = 0 #initialize the column sum for the current column
            for i in range(len(self.M1)):  #iterate over each row in the current column and sum up the absolute values
                col_sum += abs(self.M1[i][j]) 
            max_col_sum = max(max_col_sum, col_sum)  #update the maximum absolute column sum if necessary
        return max_col_sum
    
    def normL2(self):
        #The L2 norm of a matrix is defined as the square root of the sum of the squares of all the elements in the matrix.
        
        sum_of_squares = 0

        for i in range(len(self.M1)):    #iterate through each row and column of the matrix
            for j in range(len(self.M1[i])):
                sum_of_squares += self.M1[i][j] ** 2  #square each element and add it to the sum

        l2_norm = sum_of_squares ** 0.5   #take the square root of the sum to get the L2 norm
        return l2_norm
    
    def normLinf(self):
        #The L∞ norm of a matrix is defined as the maximum absolute row sum of the matrix
        max_sum = 0
        rows = len(self.M1)
        cols = len(self.M1[0])  #all rows have the same number of columns (square matrices)

        for i in range(rows):  # Iterate over each row
            row_sum = sum(abs(self.M1[i][j]) for j in range(cols))  #compute absolute sum of elements in the current row
            max_sum = max(max_sum, row_sum)   # Update the maximum absolute row sum if necessary
        return max_sum

import time
import numpy as np
import matplotlib.pyplot as plt

    # Define test cases for matrices of size 500×500
   def generate_test_case(size):
    return np.random.rand(size, size)

# Define functions to measure time for operations on sparse and dense matrices
   def measure_time(operation, matrix):
    start_time = time.time()
    result = operation(matrix)
    end_time = time.time()
    return end_time - start_time

# Define functions to compare performance of sparse and dense matrices
   def compare_performance(matrix_size):
    dense_matrix = generate_test_case(matrix_size)
    sparse_matrix = {i: {j: np.random.rand() for j in range(matrix_size)} for i in range(matrix_size)}

    dense_time = measure_time(noLibraryMatrix.MM, dense_matrix)
    sparse_time = measure_time(SparseMatrix.sparseMM, sparse_matrix)

    return dense_time, sparse_time

# Measure performance for increasing matrix sizes
matrix_sizes = [500 * 2**i for i in range(7)]  # Starting from 500 and doubling the size until reaching computational limits
dense_times = []
sparse_times = []

  for size in matrix_sizes:
    dense_time, sparse_time = compare_performance(size)
    dense_times.append(dense_time)
    sparse_times.append(sparse_time)

# Plot the performance comparison results
plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes, dense_times, marker='o', label='Dense Matrix')
plt.plot(matrix_sizes, sparse_times, marker='o', label='Sparse Matrix')
plt.xlabel('Matrix Size')
plt.ylabel('Time (s)')
plt.title('Performance Comparison of Sparse and Dense Matrices')
plt.legend()
plt.grid(True)
plt.show()



#PART 3 NOW
#Compute eigenvalues:
#A - Iλ
#Find |A-Iλ|=0 (determinant of A-Iλ = 0)
#Calculate possible values of λ, which are the eigenvalues of A
    """def eigenvalues(self):
        num_rows = len(self.M1)
        if num_rows == 0:
            return False  # Empty matrix is not square
        num_cols = len(self.M1[0])  # Assuming all rows have the same number of columns
        if num_rows != num_cols:
            raise ValueError("The matrix is not square")
        else:
            identity = []
            for i in range(num_rows):
                row = []
                for j in range(num_rows):
                    if i == j:
                        row.append(λ)
                    else:
                        row.append(0)
                identity.append(row)
            subtracted_matrix = self.SUB(identity)
            return subtracted_matrix
            """

    def transpose(self):
        return [list(row) for row in zip(*self.M1)]   #transpose the matrix by unpacking each row and creating a new list of lists

    def power_iteration(self, num_iterations=1000):
        b_k = [1.0] * len(self.M1[0])  # Assuming column size for initial vector   #initialize a vector with all elements as 1.0 of the same length as columns in M1
        
        for _ in range(num_iterations):   #iterate over a specified number of iterations
            b_k1 = [sum(row[j] * b_k[j] for j in range(len(self.M1[0]))) for row in self.M1]  #multiply the matrix by the vector to get the next vector
            norm = sum(x**2 for x in b_k1) ** 0.5   #normalize the vector to prevent overflow or underflow
            b_k = [x / norm for x in b_k1]
        
        approx_eigenvalue = sum(  #compute the approximate eigenvalue using the vector
            sum(self.M1[i][j] * b_k[j] for j in range(len(self.M1[0]))) * b_k[i] 
            for i in range(len(self.M1))
        )
        
        return b_k, approx_eigenvalue   #return the dominant eigenvector and its corresponding eigenvalue


    def compute_svd(self, num_iterations=1000):
        AT = self.transpose()  #transpose the matrix
        ATA = noLibraryMatrix(AT).MM(self.M1)   #matrix product ATA = AT * A
        eigen_vector, approx_eigenvalue = self.power_iteration(num_iterations=num_iterations)  #power iteration to find the dominant eigenvector and its corresponding eigenvalue
        sigma = approx_eigenvalue ** 0.5   #compute the singular value sigma from the square root of the eigenvalue
        
        Sigma = [sigma]   #construct the approximate singular value matrix Sigma
        U = [eigen_vector] #construct the approximate left singular vector matrix U with the found eigenvector
        V = [eigen_vector] #construct the approximate right singular vector matrix V with the found eigenvector (does not fully represent V in SVD as V is not transposed)

        return U, Sigma, V

class DenseMatrix(noLibraryMatrix): #new class inheriting from the noLibraryMatrix class
    def __init__(self, M1):
        noLibraryMatrix.__init__(self, M1)  # call constructor of the parent class 

    def gaussian_elimination(self, vector): #perform Gaussian elimination on the matrix
        n = len(self.M1)  #get the number of rows in matrix
        self.M1 = list(map(list, self.M1))   #convert the matrix from tuple to list
        vector = list(vector)  #convert the vector from tuple to list
        
        for i in range(n):  #iterate over rows for partial pivoting and elimination
            max_row = max(range(i, n), key=lambda x: abs(self.M1[x][i]))  #partial pivoting: find the row with the maximum absolute value in the current column
            self.M1[i], self.M1[max_row] = self.M1[max_row], self.M1[i]   #swap the current row with the row containing the maximum element
            vector[i], vector[max_row] = vector[max_row], vector[i]
            for j in range(i+1, n):   #eliminate entries below the pivot element
                factor = self.M1[j][i] / self.M1[i][i]   #compute the factor by which the row should be multiplied for elimination
                for k in range(i, n):   #row operations to eliminate the element below the pivot
                    self.M1[j][k] -= factor * self.M1[i][k]
                vector[j] -= factor * vector[i]   #back substitution on the vector
        
        #back substitution to find the solution vector
        x = [0 for _ in range(n)]  #initialize the solution vector
        for i in range(n-1, -1, -1):  #iterate backwards through the rows
            x[i] = vector[i]  #assign the value of the vector element
            for j in range(i+1, n):   #back substitution to solve for each variable
                x[i] -= self.M1[i][j] * x[j]  #subtract the known terms
            x[i] /= self.M1[i][i]  #divide by the coefficient of the variable in the equation
        return x

class SparseMatrix(noLibraryMatrix):  # new class inheriting from noLibraryMatrix
    def __init__(self, M1):
        noLibraryMatrix.__init__(self, M1)
        self.M1 = M1 

    def sparseMM(self, other):  #sparse matrix multiplication
        if len(self.M1[0]) != len(other):  #check if the number of columns in the first matrix matches the number of rows in the second matrix
            raise ValueError("Matrices dimensions are not compatible for multiplication")

        result = []  #initialize the result list
        for i in range(len(self.M1)):   #iterate over rows of the first matrix
            row_result = []  #initialize the result for the current row
            for j in range(len(other[0])):   #iterate over columns of the second matrix
                element = sum(self.M1[i][k] * other[k][j] for k in range(len(other))) #compute the element of the resulting matrix at position (i, j) by summing the product of corresponding elements from the two matrices
                if element != 0:  #if the computed element is not zero, append it to the result list
                    row_result.append((i, j, element))
            result.append(row_result) #append the result for the current row to the overall result list
        return result
        
    def jacobi_method(self, vector, matrix, iterations=1000, tolerance=1e-10):  #solve linear systems of equations (Jacobi iterative method)
        n = len(vector)  #get the size of the vector (number of equations)
        x = [0 for _ in range(n)]  #initialize the solution vector with zeros
        x_new = x.copy()  #create a copy of the solution vector for iteration

        for _ in range(iterations): #iterate over a specified number of iterations
            for i in range(n):  #iterate over each equation in the system
                sum_val = 0  #initialize the sum of other terms in the equation
                for j in range(n):    #iterate over each coefficient in the equation
                    if i != j:  #if the coefficient is not the diagonal element add the corresponding term to the sum
                        sum_val += matrix.get((i, j), 0) * x[j]
                x_new[i] = (vector[i] - sum_val) / matrix.get((i, i), 1) #compute the new value of the variable 
            if all(abs(x_new[i] - x[i]) < tolerance for i in range(n)):   #check for convergence by comparing the difference between the new and old solution vectors to a specified tolerance
                break  #exit the loop if convergence criteria are met
            x = x_new.copy()  #update the solution vector for the next iteration
        return x  







'''matrix_values = [[1, 2, 3],
                 [4, 5, 6]]

matrix_values2 = [[2, 0],
                  [0, 1],
                  [0, 1]]

matrix_values3 = [[1, 2],
                 [4, 5]]

test = noLibraryMatrix(matrix_values3)

norm = test.eigenvalues()

print(norm)'''
