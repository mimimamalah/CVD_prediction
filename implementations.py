import numpy as np
import matplotlib as plt
import Logistics


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1. / 2. * np.mean(e**2)


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        w: shape=(D,1). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N,1)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,1). The vector of model parameters.

    Returns:
        An numpy array of shape (D,1) (same shape as w), containing the gradient of the loss at w.
    """

    e = y - np.dot(tx, w)
    return -1. / (y.shape[0]) * np.dot(tx.T, e), e


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N,1)
        tx: shape=(N,D)
        w: shape=(D,1). The vector of model parameters.

    Returns:
        An array of shape (D,1) (same shape as w), containing the stochastic gradient of the loss at w.
    """


    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err



def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm using Mean Square error Loss.

    Args:
        y: numpy array of shape=(N,1)
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        w: The lastparameter w of shape (D,1),
    """

    w = initial_w.reshape(-1,1)
    y = y.reshape(-1,1)

    for n_iter in range(max_iters):
        grad, err = compute_gradient(y, tx, w)

        w = w - gamma * grad

    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (GD) algorithm using Mean Square error Loss.

    By default , Batch_size is 1

    Args:

        y: numpy array of shape=(N,1)
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        w: The lastparameter w of shape (2, ),
    """

    w = initial_w.reshape(-1,1)
    y = y.reshape(-1,1)
    loss = compute_loss(y, tx, w)

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss

    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: Mean square error obtain with the returned weight .


    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)


    a_pseudo_inv = np.linalg.pinv(a)
    w = a_pseudo_inv.dot(b)
    mse = compute_loss(y, tx, w)
    return w, mse
    




def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,1), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_:scalar :  regularisation parameter

    Returns:
        w: optimal weights, numpy array of shape (D,1), D is the number of features.
        loss : scalar
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(
        y, tx, w
    )  # Moreover, the loss returned by the regularized methods (ridge regression and reg logistic regression) should not include the penalty term.

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform Regularized Logisitc regression for @max_iters iterations

    Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        lambda_:scalar :  regularisation parameter
        initial_w: numpy array of shape=(D,1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: shape=(D, 1)
        loss: scalar number

    """

    return Logistics.reg_logistic_regression(
        y, tx, lambda_, initial_w, max_iters, gamma
    )


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform Logisitc regression for @max_iters iterations

    Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        initial_w: numpy array of shape=(D,1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: shape=(D, 1)
        loss: scalar number

    """
    return Logistics.logistic_regression(y, tx, initial_w, max_iters, gamma)
