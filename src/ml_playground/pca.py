


###########################################################################
### example for appliying PCA for variance decomposition and reconstruction
import numpy as np

# Define the matrix sigma
A = np.array([[1, 0, 1, 1, 1],
              [0, 1, 0, 0, 1],
              [1, 1, 1, 0, 1]])

# Compute SVD, SVD generalizes to rectangular matrix 
'''
https://cs357.cs.illinois.edu/textbook/notes/pca.html 
when A is a matrix of size m x n 
by default, numpy.linalg.svd(A) returns (U,s,Vh), where

U is a m x m matrix of the left singular vectors of matrix A in decending order, represents the directions in the row space of A 

s is a 1D array of singular values of sigma (not a full matrix) in a descending order, represents the "importance" or magnitude of each singgular pair. In PCA, U @ diag(s) gives the data's coordinates in the principal component space. 

Vh is a n x n matrix the transpose of the right singular vectors (or V^T) in decending order also.
The rows of Vh are the eigenvectors of A^HA and the columns of U are the eigenvectors of AA^H. the principle components or the direction of variance are rows of Vt
U and Vh are unitary (U@U^T=U^T@U=I), orthogonal matrices. 
Diagnoal matrix: rescale/stretching each axis
Orthogonal matrix: Rotation 
'''
U, s, Vh = np.linalg.svd(A)


'''
To approximate A using 
U[:,:k] @ diag(s[:k]) @ Vt[:k,:]
To restruct A using A = U @ np.diag(s) @ Vh
'''

# S has the same dimension of A. The values on the diagonal are the singular values of matrix A arranged in a descending order, evey other entry is zero.
S = np.zeros(A.shape)  
# Place singular values on the diagonal, 
for i in range(len(s)):
    S[i, i] = s[i]
S[0, 0] = s[0]

'''
to reconstruct A (or an approximation) using 
U[:,:k] @ diag(s[:k]) @ Vt[:k,:]
'''
reconstructed = U @ S @ Vh

with np.printoptions(precision=4, suppress=True):
    print("\nReconstructed matrix:\n", reconstructed)

###########################################################################
### example for appliying PCA for dimensionality reduction
import numpy as np

np.random.seed(42)
data = np.random.randn(100, 5)  # 100x5 matrix

# Standardize the data (zero mean, unit variance)
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Perform SVD
U, s, Vt = np.linalg.svd(data, full_matrices=False)

# Choose number of components to keep
n_components = 2

# Reduce dimensionality. Project data onto the top n_components singular vectors
reduced_data = np.dot(U[:, :n_components], np.diag(s[:n_components]))

print(f"Original data shape:\t{data.shape}\n"
      f"Reduced data shape:\t{reduced_data.shape}")

# Calculate explained variance ratio
explained_variance = S**2 / np.sum(S**2)
print(f"Explained variance ratio (first {n_components} components):\t"
      f"{np.sum(explained_variance[:n_components]):.4f}")





