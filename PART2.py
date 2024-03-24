#1
# Example usage
square_matrix = [
    [1, 6, 3],
    [7, 5, 8],
    [4, 9, 2]
]

non_square_matrix = [
    [6, 3, 7],
    [1, 4, 2]
]

#2
# Example usage
# Generate a random non-square matrix with more rows than columns
random_matrix = Matrix.generate_random(10, 3)  # For example, 10 rows and 3 columns
print("Original Matrix:")
random_matrix.display()

# Transform this matrix into a binary matrix
random_matrix.to_binary()
print("\nBinary Matrix:")
random_matrix.display()

#3

np.random.seed(0)
binary_matrix = np.random.randint(2, size=(100, 50))


import numpy as np  
from scipy.linalg import svd as scipy_svd  
from numpy.linalg import svd as numpy_svd  
import time  
import matplotlib.pyplot as plt 

class Matrix:  #define a class named Matrix
    def __init__(self, data): 
        self.data = data  
    
    @staticmethod  #decorator to define a static method
    def generate_random(rows, columns): 
        data = np.random.randint(10, size=(rows, columns)).tolist()    #generate a non-square matrix with random data
        return Matrix(data)  
    
    def to_binary(self, threshold=5):  #method to convert matrix data to binary
        binary_transform = lambda x: 1 if x > threshold else 0    #transform the matrix to binary using a lambda function
        self.data = [[binary_transform(item) for item in row] for row in self.data]  #apply binary transformation to each element
    
    def display(self):  #method to display matrix
        for row in self.data:  #iterate over each row in the matrix
            print(row)  

    def svd_scipy_movie(self, binary_matrix):  #method to perform SVD using SciPy
        start_time = time.time() 
        U_scipy, s_scipy, Vt_scipy = scipy_svd(binary_matrix, full_matrices=False)  #perform SVD using SciPy
        scipy_time = time.time() - start_time  #calculate elapsed time

    def svd_numpy_movie(self, binary_matrix):  #method to perform SVD using NumPy
        start_time = time.time() 
        U_numpy, s_numpy, Vt_numpy = numpy_svd(binary_matrix, full_matrices=False)  #perform SVD using NumPy
        numpy_time = time.time() - start_time  #calculate elapsed time

    def comparisontime(self):  #method to compare time between SciPy and NumPy SVD
        if scipy_time < numpy_time:  #compare elapsed time for SciPy and NumPy
            s = s_scipy  
            U = U_scipy 
            Vt = Vt_scipy  
        else:  #if NumPy SVD is faster or equally fast
            s = s_numpy  
            U = U_numpy  
            Vt = Vt_numpy  
        return U, s, Vt  

    def plot_singular_values(self):  #method to plot singular values
        plt.figure(figsize=(10, 6))  #create a new figure
        plt.plot(s, 'bo-')  #plot singular values
        plt.title('Singular Values of the Ratings Matrix')  #set title of the plot
        plt.xlabel('Singular Value Index')  #set x-axis label
        plt.ylabel('Singular Value Magnitude')  #set y-axis label
        plt.yscale('log')  #use logarithmic scale for y-axis
        plt.grid(True)  #display grid lines
        plt.show()  #show the plot

    @staticmethod  #decorator to define a static method
    def reduce_matrices(U, s, Vt, k):  #static method to reduce matrices
        #reduce dimensions
        U_k = U[:, :k]  #select first k columns of U
        S_k = np.diag(s[:k])  #select first k singular values as diagonal matrix
        Vt_k = Vt[:k, :]  #select first k rows of Vt
    
        #print the reduced matrices
        print("Reduced U (U_k):") 
        print(U_k)  
        print("\nReduced V^T (Vt_k):") 
        print(Vt_k) 

    def recommend(self, liked_movie_index, VT, selected_movies_num):  #method to recommend movies
        recommendations = []  #initialize list for recommendations
        for i in range(len(VT.T)):  #iterate over each movie's features
            if i != liked_movie_index:  #exclude liked movie index
                similarity = np.dot(VT[:, liked_movie_index], VT[:, i])  #compute cosine similarity
                recommendations.append((i, similarity))  #append movie index and similarity to recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)  #sort recommendations by similarity
        final_recommend_list = [rec[0] for rec in recommendations[:selected_movies_num]]  #select top movies
        return final_recommend_list  #return list of recommended movies

