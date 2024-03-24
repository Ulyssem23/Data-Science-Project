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


#3


# Assume binary_matrix is your ratings matrix
# For demonstration, let's create a binary matrix as an example
np.random.seed(0)
binary_matrix = np.random.randint(2, size=(100, 50))

# SciPy SVD
start_time = time.time()
U_scipy, s_scipy, Vt_scipy = scipy_svd(binary_matrix, full_matrices=False)
scipy_time = time.time() - start_time

# NumPy SVD
start_time = time.time()
U_numpy, s_numpy, Vt_numpy = numpy_svd(binary_matrix, full_matrices=False)
numpy_time = time.time() - start_time

print(f"SciPy SVD computation time: {scipy_time:.4f} seconds")
print(f"NumPy SVD computation time: {numpy_time:.4f} seconds")

# Choosing the faster method for further operations
if scipy_time < numpy_time:
    s = s_scipy
    print("Using SciPy SVD for further operations.")
else:
    s = s_numpy
    print("Using NumPy SVD for further operations.")



plt.figure(figsize=(10, 6))
plt.plot(s, 'bo-')
plt.title('Singular Values of the Ratings Matrix')
plt.xlabel('Singular Value Index')
plt.ylabel('Singular Value Magnitude')
plt.yscale('log')  # Use logarithmic scale to better visualize the drop-off
plt.grid(True)
plt.show()



#4
import numpy as np

# Example SVD results
U, s, Vt = np.linalg.svd(binary_matrix, full_matrices=False)

# Choose k based on your criterion (e.g., retaining 90% of variance)
k = 10  # Example value, choose based on your analysis

# Reduce dimensions
U_k = U[:, :k]
S_k = np.diag(s[:k])
Vt_k = Vt[:k, :]

# Print the values of U_k and Vt_k
print("Reduced U (U_k):")
print(U_k)
print("\nReduced V^T (Vt_k):")
print(Vt_k)



#5

Algorithm 1: Recommendation Algorithm
Input: liked_movie_index, VT, selected_movies_num
Output: final_recommend_list

1 Function recommend(liked_movie_index, VT, selected_movies_num):
2     recommendations = []
3     for i in range(len(VT.T)):  # Iterate over the movies in VT
4         if i != liked_movie_index:
5             # Calculate the dot product for similarity
6             similarity = dotProduct(VT[:, liked_movie_index], VT[:, i])
7             recommendations.append((i, similarity))
8     # Sort movies based on similarity in descending order
9     recommendations.sort(key=lambda x: x[1], reverse=True)
10    # Select top N movies based on the 'selected_movies_num'
11    final_recommend_list = [rec[0] for rec in recommendations[:selected_movies_num]]
12    return final_recommend_list


#le code
import numpy as np

def recommend(liked_movie_index, VT, selected_movies_num):
    recommendations = []
    for i in range(len(VT.T)):  # Accessing each movie's features
        if i != liked_movie_index:
            similarity = np.dot(VT[:, liked_movie_index], VT[:, i])
            recommendations.append((i, similarity))
    recommendations.sort(key=lambda x: x[1], reverse=True)
    final_recommend_list = [rec[0] for rec in recommendations[:selected_movies_num]]
    return final_recommend_list

# Example usage assumes VT is already defined and liked_movie_index is known
# liked_movie_index = ...
# VT = ...  # Obtained from SVD
# selected_movies_num = 2
# recommendations = recommend(liked_movie_index, VT, selected_movies_num)
# print("Recommended Movie Indices:", recommendations)
