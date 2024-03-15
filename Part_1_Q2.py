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

        #iterate through each row and column of the matrix
        for i in range(len(self.M1)):
            for j in range(len(self.M1[i])):
                #square each element and add it to the sum
                sum_of_squares += self.M1[i][j] ** 2

        #take the square root of the sum to get the L2 norm
        l2_norm = sum_of_squares ** 0.5
        return l2_norm
    
    def normLinf(self):
        #The L∞ norm of a matrix is defined as the maximum absolute row sum of the matrix
        max_sum = 0
        rows = len(self.M1)
        cols = len(self.M1[0])  # Assuming all rows have the same number of columns

        for i in range(rows):
            row_sum = sum(abs(self.M1[i][j]) for j in range(cols))
            max_sum = max(max_sum, row_sum)

        return max_sum
        
class DenseMatrix(noLibraryMatrix): #create a new child class
    def __init__(self, M1):
        noLibraryMatrix.__init__(M1)

class SparseMatrix(noLibraryMatrix): #create a new  child class
    def __init__(self, M1):
        noLibraryMatrix.__init__(self, M1)

    def MM(self, other):   #matrix multiplication
        if len(self.M1[0]) != len(other):   #check if the dimensions of the matrices are compatible
            raise ValueError("Matrices dimensions are not compatible for multiplication.")
        
        result = []
        for i in range(len(self.M1)):
            row_result = []
            for j in range(len(other[0])):
                element = sum(self.M1[i][k] * other[k][j] for k in range(len(other)))
                if element != 0:
                    row_result.append((i, j, element))  # Store (row, column, value)
            result.append(row_result)
        return result
