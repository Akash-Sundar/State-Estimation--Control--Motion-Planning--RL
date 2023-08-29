import numpy as np
A = np.array([1, 2, 0, -2, -1, 0]).reshape((-1,1))
B = np.array([1, 1, -1, -1, 0, 0]).reshape((-1,1))

print(np.linalg.inv(A @ A.T) @ A.T @ B)
