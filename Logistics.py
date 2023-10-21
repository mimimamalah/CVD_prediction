import numpy as np
import matplotlib as plt


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """

    t= np.clip(t,-700,700)
    return 1.0 / (1.0 + np.exp(-t))


def calculate_gradient(y, tx, w):
    """compute the gradient of loss for the negative log likelyhood loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    N = y.shape[0]
    pred = sigmoid(tx.dot(w))
    grad = 1 / N * tx.T.dot(pred - y)
    return grad


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    N = y.shape[0]
    inner_prod = tx.dot(w)



    sig_prod = sigmoid(inner_prod)


    #Assert small constant to avoid logarithmic instability 
    epsilon = 1e-15
    return -1 / N * (y * np.log(sig_prod + epsilon) + (1 - y) * np.log(1 - sig_prod+epsilon)).sum()


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """

    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, grad


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """

    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)

    """
    grad = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)
    w = w - grad * gamma
    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):

    #print("initial w : ",initial_w)
    print("uwu")



    losses = []
    w = initial_w
    loss = calculate_loss(y, tx, w)
    # start the logistic regression
    for iter in range(max_iters):
        
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration=={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
    # visualization
    return w, calculate_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []

    w = initial_w
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    nonPenalizedLoss = calculate_loss(y, tx, w)
    return w, nonPenalizedLoss
