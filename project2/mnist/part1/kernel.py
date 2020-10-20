import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    prod = np.matmul(X,Y.transpose())
    return (prod + c ) ** p
    # YOUR CODE HERE
    raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    n,d = X.shape
    m,d2 = Y.shape

    X_new = np.broadcast_to (X[...,None],[n,d,m])
    Y_new = np.broadcast_to (Y[...,None],[m,d,n])
    Y_new_T = Y_new.transpose(2,1,0)
    diff = X_new - Y_new_T
    diff_norm = np.linalg.norm(diff, axis = 1)
    return np.exp(-gamma * (diff_norm**2))
    raise NotImplementedError
