from Part_2 import Matrix

import numpy as np
import sys

choice = int(sys.argv[1])

matrixx = Matrix(np.random.randint(0, 101, size=(100, 100)))

if choice == 1:
    print(Matrix.generate_random(5,10))
    
if choice == 2:
    print(matrixx.to_binary())
    
if choice == 3:
    matrixx.display()

if choice == 4:
    U, S, vT = matrixx.svd_comparison()

if choice == 5:
    print(U, S, vT)
    
if choice == 6:
    matrixx.reduce_matrices(U, S, vT, 10)

if choice == 7:
    print(matrixx.recommend(1, vT, 1))
