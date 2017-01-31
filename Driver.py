import numpy as np
from scipy import io as sio
from Plotter import Plotter
from Utils import MatrixUtils as Utils
import scipy.optimize
from GradientDescentSolver import GradientDescentSolver as solver

np.set_printoptions(suppress=True)

# load & format the training data
data_raw = sio.loadmat('C:\\Users\\Matthew\\Documents\\Learning Resources\\Projects\\EX4\\ex4data1')
X = data_raw['X']
y = data_raw['y'][:, 0]

(m, n) = X.shape

input_layer_size = n
hidden_layer_size = 25
num_labels = 10
num_layers = 3
lambda_ = 1

# Plotter.display(X, random_x=True)

# load pre-made Theta arrays
ex4weights = scipy.io.loadmat('C:\\Users\\Matthew\\Documents\\Learning Resources\\Projects\\EX4\\ex4weights.mat')
Theta1 = ex4weights['Theta1']
Theta2 = ex4weights['Theta2']

# unroll parameters
nn_params = np.concatenate((Theta1.flat, Theta2.flat))

# compute cost for example
(J, grad) = solver.nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels,
                                    X, y, lambda_, num_layers)
print(J)

scipy.optimize.fmin_cg(nn_cost_function, )