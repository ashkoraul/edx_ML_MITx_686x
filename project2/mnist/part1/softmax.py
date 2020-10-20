import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    n, d = X.shape
    k = theta.shape[0]
    H = np.zeros([k,n], dtype=np.float64)
    for i in range(n):
        #theta_x_by_tau = np.square(np.linalg.norm(X[i] * theta, axis = 1))/temp_parameter # compute the theta_j .x / temperature param
        theta_x_by_tau = np.matmul(theta, X[i]) / temp_parameter
        theta_x_by_tau_withc = theta_x_by_tau-theta_x_by_tau.max()
        exp_theta_x_by_tau_withc = np.exp(theta_x_by_tau_withc)
        H[:, i] = 1/exp_theta_x_by_tau_withc.sum() * exp_theta_x_by_tau_withc
    return H


    #YOUR CODE HERE
    raise NotImplementedError

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    n, d = X.shape
    k = theta.shape[0]
    H_new = np.zeros([k, n], dtype=np.float64)
    j = range(k)
    # for i in range(n):
    #     theta_x_by_tau = np.matmul(theta, X[i]) / temp_parameter
    #     exp_theta_x_by_tau = np.exp(theta_x_by_tau)
    #
    #     #if ((Y[i]==i) ):
    #     H[:, i] = (Y[i] ==j) * np.log(exp_theta_x_by_tau/exp_theta_x_by_tau.sum())
    H = compute_probabilities(X, theta, temp_parameter)

    # logH= np.log(H)
    # for i in range(n):
    #     H[:, i] = (Y[i] == j) * logH[:, i]
    # theta_sq = theta * theta
    # c = -1/n * H.sum() + lambda_factor/2 * theta_sq.sum()

    for i in range(n):
        H_new[Y[i],i] = np.log(H[Y[i], i ])
    theta_sq = theta * theta
    c_new = -1 / n * H_new.sum() + lambda_factor / 2 * theta_sq.sum()


    return c_new
    #YOUR CODE HERE
    raise NotImplementedError

def run_gradient_descent_iteration_AR(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    n, d = X.shape
    k = theta.shape[0]
    # H_new = np.zeros([k, n], dtype=np.float64)
    H = compute_probabilities(X, theta, temp_parameter)
    j = range(k)
    # theta_sum = np.zeros([k,d], dtype=np.float64)
    # # H_new = H
    # # H[H<0] = 0.0
    # # H[H>1] = 1.0
    # for i in range(n):
    #     term1 = X[i].reshape(1, d)
    #     term2 = ((Y[i] == j) - H[:, i])
    #     term2 = term2.reshape(k, 1)
    #
    #     theta_sum = theta_sum + term1*term2
    theta_sum = np.zeros([k, d], dtype=np.float64)
    M = sparse.coo_matrix(([1] * n, (Y, range(n))), shape=(k, n), dtype=np.float64).toarray()
    diff = M-H

    for i in range(n):
        term1 = X[i].reshape(1, d)
        term2 = diff[:, i]
        term2 = term2.reshape(k, 1)
        theta_sum = theta_sum + term1 * term2

    return theta - alpha*(-1/(temp_parameter*n)*theta_sum + lambda_factor*theta)

    raise NotImplementedError

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    n, d = X.shape
    k = theta.shape[0]
    H = compute_probabilities(X, theta, temp_parameter)
    theta_sum = np.zeros([k,d], dtype=np.float64)
    M = sparse.coo_matrix(([1] * n, (Y, range(n))), shape=(k, n), dtype=np.int8).toarray()
    diff = M-H
    theta_sum = np.matmul(diff, X)
    return theta - alpha*(-1/(temp_parameter*n)*theta_sum + lambda_factor*theta)

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    return train_y % 3, test_y % 3
    #YOUR CODE HERE
    raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)


    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    y_estimate = get_classification(X, theta, temp_parameter) % 3
    error = (y_estimate != Y)
    return error.sum()/Y.shape[0]
    #YOUR CODE HERE
    raise NotImplementedError

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
