import numpy as np
from Utils import MatrixUtils


class GradientDescentSolver:

    @staticmethod
    def _back_propagate(activation_vals, z_vals, thetaList, num_layers, y, g):
        # initial delta
        di = activation_vals[-1] - y
        delta = [di]

        # calculate delta values
        for i in range(num_layers-1, 1, -1):
            di = delta[0].dot(thetaList[i-1])
            di = np.multiply(di[:, 1:], g(z_vals[i-1]))
            delta.insert(0, di)

        big_delta = []
        for i in range(0, num_layers-1):
            ai = MatrixUtils.add_ones_column(activation_vals[i])
            big_delta.insert(i, np.transpose(delta[i]).dot(ai))

        return big_delta

    @staticmethod
    def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_, num_layers):
        """ NNCOSTFUNCTION Implements the neural network cost function for a neural network which performs classification
           [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
           X, y, lambda) computes the cost and gradient of the neural network. The
           parameters for the neural network are "unrolled" into the vector
           nn_params and need to be converted back into the weight matrices.

           The returned parameter grad should be a "unrolled" vector of the
           partial derivatives of the neural network.
        """

        # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
        # for our 2 layer neural network
        t1_len = (input_layer_size + 1) * hidden_layer_size
        Theta1 = nn_params[:t1_len].reshape(hidden_layer_size, input_layer_size + 1)
        Theta2 = nn_params[t1_len:].reshape(num_labels, hidden_layer_size + 1)
        m = X.shape[0]

        # reformat output column y s.t. it is broken into K distinct classes
        Y = MatrixUtils.classify(y)

        # organize theta matrices into an array
        theta_list = [Theta1, Theta2]

        # perform feed forward algorithm to get activation values for each layer of the neural network
        (activation_vals, z_vals) = GradientDescentSolver._feed_forward(theta_list, num_layers, X, MatrixUtils.sigmoid)
        hx = activation_vals[-1]

        # calculate the regularized cost function
        J_reg = (lambda_/(2*m)) * GradientDescentSolver._regularization_term(theta_list, num_layers)
        J = (1/m) * np.sum((-Y * np.log(hx) - (1 - Y) * np.log(1 - hx))) + J_reg

        # perform back propagation algorithm to get the gradients for each value of theta
        delta_vals = GradientDescentSolver._back_propagate(activation_vals, z_vals, theta_list, num_layers,
                                                           Y, MatrixUtils.sigmoid_gradient)

        Theta_grad_list = []
        for i in range(0, num_layers-1):
            Theta_grad_i = (1/m) * delta_vals[i]
            Theta_grad_i[:, 1:] = Theta_grad_i[:, 1:] + (lambda_/m) * theta_list[i][:, 1:]
            Theta_grad_list.append(Theta_grad_i)

        # Unroll gradients into 1-d row vector
        gradient = MatrixUtils.unroll(Theta_grad_list)

        return J, gradient

    @staticmethod
    def _feed_forward(thetaList, num_layers, X, g):
        """A loop-based feed forward algorithm. Takes a list of theta arrays,
        outputs a list of activation matrices"""
        a = [X]
        z = [1]
        for i in range(1, num_layers):
            theta = np.transpose(thetaList[i-1])
            ai = MatrixUtils.add_ones_column(a[i-1])
            zi = ai.dot(theta)
            z.append(zi)
            a.append(g(zi))

        return (a, z)

    @staticmethod
    def _regularization_term(thetaList, num_layers):
        """A loop-based algorithm to create the regularization term
        for a neural network cost function"""
        reg_sum = 0
        for i in range(0, num_layers-1):
            theta_i = thetaList[i]
            theta_i = np.delete(theta_i, 0, 1)
            reg_sum = reg_sum + sum(sum(theta_i**2))

        return reg_sum

    @staticmethod
    def generate_theta(shape, epsilon):
        return np.random.random(shape) * 2 * epsilon - epsilon
