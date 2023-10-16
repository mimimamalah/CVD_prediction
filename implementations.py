import numpy as np
import matplotlib as plt
import Logistics 


def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1 / 2 * np.mean(e**2)

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    
    e = y - np.dot(tx,w)
    return - 1/(y.shape[0])* np.dot(tx.T,e),e

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    ### SOLUTION
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err



def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
     
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
       
        w = w - gamma * grad


        
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, w

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
        # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

      
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=1, num_batches=1
        ):
            # compute a stochastic gradient and loss
            grad, _= compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
       

        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return losses, w



def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_ ):
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = (y-tx.dot(w))+lambda_ * np.squeeze(w.T.dot(w))
    
    return w,loss







def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    return Logistics.reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return Logistics.logistic_regression(y, tx, initial_w, max_iters, gamma)
