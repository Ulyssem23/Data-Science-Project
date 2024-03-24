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

class Matrix:
    def __init__(self, data):
            self.data = data
    
    @staticmethod
    def generate_random(rows, columns):
        # Generate a non-square matrix with random data
        data = np.random.randint(10, size=(rows, columns)).tolist()
        return Matrix(data)
    
    def to_binary(self, threshold=5):
        # Transform the matrix to binary using a lambda function
        binary_transform = lambda x: 1 if x > threshold else 0
        self.data = [[binary_transform(item) for item in row] for row in self.data]
    
    def display(self):
        for row in self.data:
            print(row)

    def svd_scipy_movie(self, binary_matrix):
        start_time = time.time()
        U_scipy, s_scipy, Vt_scipy = scipy_svd(binary_matrix, full_matrices=False)
        scipy_time = time.time() - start_time

    def svd_numpy_movie(self, binary_matrix):
        start_time = time.time()
        U_numpy, s_numpy, Vt_numpy = numpy_svd(binary_matrix, full_matrices=False)
        numpy_time = time.time() - start_time

    def comparisontime(self):
    if scipy_time < numpy_time:
        s = s_scipy
    else:
        s = s_numpy

    def  plot_singular_values(self):
        plt.figure(figsize=(10, 6))
        plt.plot(s, 'bo-')
        plt.title('Singular Values of the Ratings Matrix')
        plt.xlabel('Singular Value Index')
        plt.ylabel('Singular Value Magnitude')
        plt.yscale('log')  # Use logarithmic scale to better visualize the drop-off
        plt.grid(True)
        plt.show()




import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd as scipy_svd
from numpy.linalg import svd as numpy_svd

def perform_svd_and_compare(binary_matrix):
    # Perform SVD using scipy
    start_time_scipy = time.time()
    U_scipy, s_scipy, Vt_scipy = scipy_svd(binary_matrix, full_matrices=False)
    scipy_time = time.time() - start_time_scipy

    # Perform SVD using numpy
    start_time_numpy = time.time()
    U_numpy, s_numpy, Vt_numpy = numpy_svd(binary_matrix, full_matrices=False)
    numpy_time = time.time() - start_time_numpy

    # Compare and keep the best
    if scipy_time < numpy_time:
        s = s_scipy
        U = U_scipy
        Vt = Vt_scipy
        print(f"Scipy was faster: {scipy_time}s")
    else:
        s = s_numpy
        U = U_numpy
        Vt = Vt_numpy
        print(f"Numpy was faster: {numpy_time}s")

    return U, s, Vt

def plot_singular_values(s):
    plt.figure(figsize=(10, 6))
    plt.plot(s, 'bo-')
    plt.title('Singular Values of the Ratings Matrix')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Singular Value Magnitude')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def reduce_and_print_matrices(U, s, Vt, k):
    # Reduce dimensions
    U_k = U[:, :k]
    S_k = np.diag(s[:k])
    Vt_k = Vt[:k, :]

    # Print the reduced matrices
    print("Reduced U (U_k):")
    print(U_k)
    print("\nReduced V^T (Vt_k):")
    print(Vt_k)

# Assume binary_matrix is defined
# U, s, Vt = perform_svd_and_compare(binary_matrix)
# plot_singular_values(s)
# Choose k based on your criterion
# k = 10  # Example value
# reduce_and_print_matrices(U, s, Vt, k)


#5

    def recommend(self, liked_movie_index, VT, selected_movies_num):
        recommendations = []
        for i in range(len(VT.T)):  # Accessing each movie's features
            if i != liked_movie_index:
                similarity = np.dot(VT[:, liked_movie_index], VT[:, i])
                recommendations.append((i, similarity))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        final_recommend_list = [rec[0] for rec in recommendations[:selected_movies_num]]
        return final_recommend_list

