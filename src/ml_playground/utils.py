


################################################################################
# A simple LR demo
################################################################################
# from abc import ABC # Abstract Base Classes
from typing import Any

import numpy as np
from numpy.typing import NDArray
# from pydantic import BaseModel

class LinearRegression():
   
   def __init__(self) -> None:
      """Initialize the LinearRegression"""
      self.intercept = None
      self.coefficients = None
   
   def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Any:
      """
      Fit a linear regression model.

      Parameters
      ----------
      X : np.ndarray
         an array containing feature values for training.
      y : np.ndarray
         an array containing the response values.
      
      Returns
      -------
      Any
         a trained model.
      """
      
      ones = np.ones((len(X), 1)) # y-intercept (or bias)
      X = np.hstack((ones, X))
      # (X^T*X)^(-1)*X^T*y
      XT = X.T # Transpose of X
      XTX = XT.dot(X)   # X^T*X
      # NumPy linear algebra functions to inverse a matrix.
      # also scipy.linalg.inv(a) in SciPy library 
      XTX_inv = np.linalg.inv(XTX)  # (X^T*X)^(-1), inverse
      XTy = XT.dot(y)   # X^T*y
      self.coefficients = XTX_inv.dot(XTy)   # Calculate the coefficients
 
      
   def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
      """
      Make predictions from a model.

      Parameters
      ----------
      X : np.ndarray
         an array containing feature values for prediction.
      
      Returns
      -------
      np.array
         a numpy array of predictions.
      """
      ones = np.ones((len(X), 1)) # intercept
      X = np.hstack((ones, X))
      # Apply the coefficients to make predictions. numpy.matmul(A, B) or A @ B or A.dot(B). the @ operator is preferable.
      # return X.dot(self.coefficients)
      return X @ self.coefficients
   
   # Calculate R-squared (the coefficient of determination)
   def Rsquared(self, X:NDArray[np.float64], y:NDArray[np.float64]) -> float:
      """
      Make predictions from a model.

      Parameters
      ----------
      X : np.ndarray
         an array containing feature values for prediction.
      y : np.ndarray
         an array containing the response values.
      
      Returns
      -------
      float
         the calculated R-squared value.
      """
      ypred = self.predict(X)
      ss_total = np.sum((y - np.mean(y))**2) # Total sum of squares
      ss_residual = np.sum((y - ypred)**2)   # Residual sum of squares
      return 1 - ss_residual / ss_total   # R-squared
   

from sklearn import datasets

# Generate some toy data
X, y = datasets.make_regression(
        n_samples=500,
        n_features=1, 
        # the code is generalized for any number (n > 1) of features
        noise=15,
        random_state=4
        )

# Initialize and fit a model
model = LinearRegression()
model.fit(X, y)
#print(model.coefficients)

#print(model.intercept)
dir(model)

# Make prediction
y_pred = model.predict(X)

# R-squared value
print(model.Rsquared(X, y))

# one-D plot to illustrate how well the model fits the data
import matplotlib.pyplot as plt
# select one feature only for one-D plot
feature_index = 0

intercept = model.coefficients[0]

fig, ax = plt.subplots(figsize=(8,6))

# generate x values and y = a*x + b for line plot
x_lp = np.linspace(X[:,feature_index].min(), X[:,feature_index].max(), 100)
y_lp = model.coefficients[feature_index+1]*x_lp + intercept

ax.scatter(X[:,feature_index], y, color='blue')
ax.scatter(X[:,feature_index], y_pred, color='red')
ax.plot(x_lp, y_lp, color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Linear Regression')
plt.show() # plt.savefig()


################################################################################
# XGBoost CPU Utilization (Number of estimators) vs. Number of Estimators and Threads
# https://xgboosting.com/xgboost-cpu-usage-below-100-during-training/
################################################################################

import psutil
import pandas as pd
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Generate a synthetic dataset for binary classification
X, y = make_classification(n_samples=1000000, n_features=20, n_informative=10, n_redundant=5, random_state=42)

# Define the range of threads and estimators to test
threads_range = range(1, 5)
estimators_range = [10, 50, 100, 200, 300, 400, 500]

# Initialize a DataFrame to store the results
results_df = pd.DataFrame(columns=['threads', 'estimators', 'cpu_utilization'])

# Iterate over the number of threads and estimators
for threads in threads_range:
   for estimators in estimators_range:
       # Initialize an XGBClassifier with the specified parameters
       model = XGBClassifier(n_jobs=threads, n_estimators=estimators, random_state=42)

       # Measure CPU utilization before training
       _ = psutil.cpu_percent()

       # Train the model
       model.fit(X, y)

       # Measure CPU utilization since last call
       cpu_percent_during = psutil.cpu_percent()

       result = pd.DataFrame([{
                           'threads': threads,
                           'estimators': estimators,
                           'cpu_utilization': cpu_percent_during
                       }])
       # Report progress
       print(result)

       # Append the results to the DataFrame
       results_df = pd.concat([results_df, result], ignore_index=True)

# Pivot the DataFrame to create a matrix suitable for plotting
plot_df_cpu = results_df.pivot(index='estimators', columns='threads', values='cpu_utilization')

# Create a line plot
plt.figure(figsize=(10, 6))
for threads in threads_range:
   plt.plot(plot_df_cpu.index, plot_df_cpu[threads], marker='o', label=f'{threads} threads')

plt.xlabel('Number of Estimators')
plt.ylabel('CPU Utilization (%)')
plt.title('XGBoost CPU Utilization vs. Number of Estimators and Threads')
plt.legend(title='Threads')
plt.grid(True)
plt.xticks(estimators_range)
plt.show()

################################################################################
# Tune XGBoost "alpha" Parameter
# Applying matplotlib.pyplot.semilogx to make a plot with log scaling on the x-axis
# Applying matplotlib.pyplot.fill_between to plot confidence interval
# https://xgboosting.com/tune-xgboost-alpha-parameter/
################################################################################
import xgboost as xgb
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score

# Create a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, noise=0.1, random_state=42)

# Configure cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameter grid
param_grid = {
    'alpha': [0, 0.01, 0.1, 1, 10, 100]
}

# Set up XGBoost regressor
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X, y)

# Get results
print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best CV R^2 score: {grid_search.best_score_:.4f}")

# Plot alpha vs. R^2 score
import matplotlib.pyplot as plt
results = grid_search.cv_results_

plt.figure(figsize=(10, 6))
plt.semilogx(param_grid['alpha'], results['mean_test_score'], marker='o', linestyle='-', color='b')
plt.fill_between(param_grid['alpha'], results['mean_test_score'] - results['std_test_score'],
                 results['mean_test_score'] + results['std_test_score'], alpha=0.1, color='b')
plt.title('Alpha vs. R^2 Score')
plt.xlabel('Alpha (log scale)')
plt.ylabel('CV Average R^2 Score')
plt.grid(True)
plt.show()

################################################################################
# Animated image using a precomputed list of images
# https://matplotlib.org/stable/gallery/animation/index.html
################################################################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()

def f(x, y):
    return np.sin(x) + np.cos(y)

x = np.linspace(0, 2 * np.pi, 120)
y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(60):
    x += np.pi / 15
    y += np.pi / 30
    im = ax.imshow(f(x, y), animated=True)
    if i == 0:
        ax.imshow(f(x, y))  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
# To save the animation, use e.g.
# ani.save("movie.mp4")
# or
# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Me'), bitrate=1800)
# ani.save("movie.mp4", writer=writer)
plt.show()


################################################################################
# Euclidean distance matrix
import numpy as np
import numpy.typing as npt

# https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/#eq1

def computer_squared_euc_dists(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute matrix containing squared euclidean distance for all pairs of points in input matrix X, a dataset consisting N data points each has D dimensions (Number of features). ||x_i - x_j||^2

    # Arguments:
        X: matrix of size NxD
    # Returns:
        NxN matrix D, with entry D_ij = squared euclidean distance between rows X_i and X_j
    """
    # Math? See https://stackoverflow.com/questions/37009647
    sum_X = np.sum(np.square(X), axis=1)
    sd = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return sd
"""
numpy.square() - the element-wise square of the input.
numpy.add() - add arguments element-wise
numpy.sum() - Sum of array elements over a given axis.
In the matrix D where a[i] is the ith row of your original matrix and 
SD[i,j]=(a[i]-a[j])(a[i]-a[j]).T = r[i] - 2 a[i]a[j].T + r[j] 
where r[i] (r[j]) is squared norm of ith (jth) row of the original matrix.
Note broadcasting is applied on the squared norm resulting in a column vector (https://numpy.org/doc/stable/user/basics.broadcasting.html)
"""

def distance_matrix(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64], squared=False)-> npt.NDArray[np.float64]:
    """
    Compute all pairwise distances between vectors in matrix A and B. A and B can be two datasets with the same dimensions(number of features), the number of rows or records may or may not be the same. 
    for example,
    A = np.array([[1,2,3],[2,3,4],[0,1,2]])     
    B = np.array([[1,2,3],[4,3,2]])

    Parameters
    ----------
    A : np.array
        shape should be (M, K)
    B : np.array
        shape should be (N, K)

    Returns
    -------
    D : np.array
        A matrix D of shape (M, N).  Each entry in D i,j represnets the
        distance between row i in A and row j in B.

    """
    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = (A*A).sum(axis=1).reshape((M,1))*np.ones(shape=(1,N))
    B_dots = (B*B).sum(axis=1)*np.ones(shape=(M,1))
    D_squared =  A_dots + B_dots -2*A.dot(B.T)

    # decide whether to return squared distance or just distance
    if squared == False:
        # numpy.less(x1, x2) check if x1 < x2 element-wise.
        zero_mask = np.less(D_squared, 0.0)
        # replace negative with 0.0
        D_squared[zero_mask] = 0.0
        # or simply apply nump.where()
        # D_squared = np.where(D_squared < 0.0, 0.0, D_squared)
        
        return np.sqrt(D_squared)

    return D_squared


A = np.array([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
B = np.array([[-2.1763, -0.4713], [-0.6986, 1.3702]])

# A more generalized version of the distance matrix is available from scipy
from scipy.spatial import distance_matrix
distance_matrix(A, B)

# torch.cdist computes batched the p-norm distance between each pair of the two collections of row vectors.
import torch
a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
# If a has shape B×P×M and b has shape B×R×M then the output will have shape B×P×R, where B is batch size.
torch.cdist(a, b, p=2)

from scipy.spatial import distance
distance.cdist(A, B, 'euclidean')

distance.cdist(A, A, 'euclidean') # distance between each pair of the two collections of inputs
distance.pdist(A, 'euclidean') # Pairwise distances between observations in n-dimensional space.

################################################################################




################################################################################
# 
################################################################################



################################################################################
# 
################################################################################



################################################################################
# 
################################################################################
