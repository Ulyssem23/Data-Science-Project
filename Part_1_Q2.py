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
        
        for i in range(len(self.M1)):
    #loop over the rows of M1
            for j in range(len(self.M2[0])):
        #loop over the columns of M2
                for k in range(len(self.M2)):
            #loop over the rows of the second matrix M2 + columns of the first matrix M1
            #compute dot product between the ith row of M1 and the kth column of M2
                    result[i][j] += self.M1[i][k] * self.M2[k][j]
           #update each element of the result matrix
        return result

    def MV(self, vector): #vector matrix multiplication
            #store the vector as an attribute of the class
            self.vector = vector
            #check if the vector is one-dimensional 
            is_one_dimensional = all(isinstance(x, (int, float)) for x in vector)
            #if the vector is not one-dimensional, raise a ValueError
            if not is_one_dimensional:
                raise ValueError("Vector's dimension is incorrect")
            #initialize a result list with zeros
            result = [0] * len(self.M1)
        
            #iterate over the rows of M1
            for i in range(len(self.M1)):
                #iterate over the elements of the vector
                for k in range(len(vector)):
                    #multiply  elements of M1 and the vector and add to the result
                    result[i] += self.M1[i][k] * vector[k]
            return result

    def ADD(self, M3): #addition
        self.M3 = M3  #store  M3 as an attribute of the class
        result = []  #initialize result list
        
        #iterate over each row of the matrices M1 and M3
        for i in range(len(self.M1)):
            row = []  #initialize a row for the result matrix
            #iterate over each column of the matrices M1 and M3
            for j in range(len(self.M1[0])):
                #add corresponding elements of M1 and M3 and append to the row
                row.append(self.M1[i][j] + M3[i][j])
            #append row to the result matrix
            result.append(row)
        
        return result 

    def SUB(self, M3): #substraction
        #store  M3 as an attribute of the class
        self.M3 = M3
        #initialize an empty list to store the result matrix
        result = []
        #iterate over the rows of M1
        for i in range(len(self.M1)):
            #initialize an empty list to store each row of the result matrix
            row = []
            #iterate over the columns of M1
            for j in range(len(self.M1[0])):
                #subtract elements of M1 and M3 and append to the row
                row.append(self.M1[i][j] - M3[i][j])
            result.append(row)
        return result

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
    
    def normL2(self):
        #

class DenseMatrix(noLibraryMatrix):
    def __init__(self, M1):
        noLibraryMatrix.__init__(M1)

class SparseMatrix(noLibraryMatrix):
    def __init__(self, M1):
        noLibraryMatrix.__init__(self, M1)

    def MM(self, other):
        # Check if the dimensions of the matrices are compatible
        if len(self.M1[0]) != len(other):
            raise ValueError("Matrices dimensions are not compatible for multiplication.")
        
        # Perform matrix multiplication
        result = []
        for i in range(len(self.M1)):
            row_result = []
            for j in range(len(other[0])):
                element = sum(self.M1[i][k] * other[k][j] for k in range(len(other)))
                if element != 0:
                    row_result.append((i, j, element))  # Store (row, column, value)
            result.append(row_result)
        return result
