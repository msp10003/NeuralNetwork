import numpy as np


class MatrixUtils:

    @staticmethod
    def add_ones_column(array):
        return np.insert(array, 0, 1, axis=1)

    @staticmethod
    def classify(col_vector):
        """Takes input vector and returns a new array of the same rows but
        with additional columns. Performs the transformation y(i) => index(y(i))"""
        K = np.unique(col_vector).size
        Y = np.zeros(shape=(col_vector.shape[0], K))
        for i in range(0, col_vector.shape[0]):
            Y[i, col_vector[i]-1] = 1

        return Y

    @staticmethod
    def sigmoid(x):
        z = 1/(1+np.exp(-x))
        return z

    @staticmethod
    def sigmoid_gradient(x):
        z = MatrixUtils.sigmoid(x) * (1 - MatrixUtils.sigmoid(x))
        return z

    @staticmethod
    def zero_ind(col_vector):
        """Method to zero index the input Y"""
        for i in range(0, col_vector.shape[0]):
            if col_vector[i] == 10:
                col_vector[i] = 0

        return col_vector

    @staticmethod
    def gradient_check(J, theta, epsilon):
        pass

    @staticmethod
    def unroll(matrix_list):
        matrix = np.empty([0, 0])
        for m in matrix_list:
            matrix = np.concatenate((matrix.flat, m.flat))

        return matrix