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
    def __len__(self):
        return len(self.M1)
    
    def __getitem__(self, key):
        return self.M1[key]

    def power_iteration(self, num_iterations=1000):
        n = len(self)
        vector = [1.0] * n  # Initial vector
        for _ in range(num_iterations):
            # Calculate the matrix-by-vector product Ab
            vector1 = [sum(self[i][j] * vector[j] for j in range(n)) for i in range(n)]
            # Calculate the norm of b_k1
            norm_vector1 = sum(x**2 for x in vector1) ** 0.5
            # Normalize the vector
            vector = [x / norm_vector1 for x in vector1]
        # Return the approximate eigenvalue
        return sum(self[i][j] * vector[j] for i in range(n) for j in range(n)) / sum(vector[i]**2 for i in range(n))
    

    def transpose(self):
        return [list(row) for row in zip(*self.M1)]

        
    def compute_svd(self, num_iterations=1000):
        AT = self.transpose()
        AAT = self.MM(AT)
        ATA = noLibraryMatrix(AT).MM(self.M1)
            
        # Compute left singular vectors (U)
        u, _ = self.power_iteration(num_iterations=num_iterations)  # Adjusted call
        U = [u]

        # Compute right singular vectors (V)
        v, sigma = self.power_iteration(num_iterations=num_iterations)  # Adjusted call
        V = [v]
        Vfinal = [list(row) for row in zip(*V)]

        # Singular values (Sigma) - approximation
        Sigma = [sigma]
        
        return U, Sigma, Vfinal


_________________________________


import Part_1_Q_2

matrixx = Part_1_Q_2.noLibraryMatrix(((1, 0), (0, 1)))

# Adjust the number of iterations as needed
print(matrixx.compute_svd(num_iterations=1000))


__________________________________

Traceback (most recent call last):
  File "c:\Users\ulyss\Documents\Visual Studio\Python\Object oriented programming\Data science project\Main.py", line 40, in <module>
    print(matrixx.compute_svd(num_iterations=1000))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\ulyss\Documents\Visual Studio\Python\Object oriented programming\Data science project\Part_1_Q_2.py", line 146, in compute_svd
    u, _ = self.power_iteration(num_iterations=num_iterations)  # Adjusted call
    ^^^^

________________________________

class noLibraryMatrix: 
    def __init__(self, M1):
        self.M1 = M1

    # Including other methods (MM, MV, ADD, SUB, etc.) here

    def transpose(self):
        return [list(row) for row in zip(*self.M1)]

    def power_iteration(self, num_iterations=1000):
        b_k = [1.0] * len(self.M1[0])  # Assuming column size for initial vector
        for _ in range(num_iterations):
            # Multiply by matrix
            b_k1 = [sum(row[j] * b_k[j] for j in range(len(self.M1[0]))) for row in self.M1]
            # Normalize
            norm = sum(x**2 for x in b_k1) ** 0.5
            b_k = [x / norm for x in b_k1]
        # Approximate eigenvalue
        approx_eigenvalue = sum(
            sum(self.M1[i][j] * b_k[j] for j in range(len(self.M1[0]))) * b_k[i] 
            for i in range(len(self.M1))
        )
        return b_k, approx_eigenvalue

    def compute_svd(self, num_iterations=1000):
        AT = self.transpose()
        ATA = noLibraryMatrix(AT).MM(self.M1)  # Correct multiplication order for ATA
        eigen_vector, approx_eigenvalue = self.power_iteration(num_iterations=num_iterations)

        # Simplify the computation for U, Sigma, V based on power iteration results
        # Note: This is a simplified approach and does not fully compute SVD for educational purposes
        sigma = approx_eigenvalue ** 0.5
        U = [eigen_vector]  # This simplification does not fully represent U in SVD
        Sigma = [sigma]
        V = [eigen_vector]  # Simplification, does not fully represent V in SVD

        return U, Sigma, V

# Example usage
matrixx = noLibraryMatrix([[1, 0], [0, 1]])
U, Sigma, V = matrixx.compute_svd(num_iterations=1000)
print("U:", U)
print("Sigma:", Sigma)
print("V^T:", V)  # Note: V is not transposed in this output for simplification
