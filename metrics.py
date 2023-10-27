import numpy as np

def calculate_parameters(y_test, y_pred):
    """Calculates the parameters for the metrics
    
    Args:
        y_test: test labels
        y_pred: predicted labels
        
    Returns:
        tp: true positive
        tn: true negative
        fp: false positive
        fn: false negative
    """
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == -1) & (y_pred == -1))
    fp = np.sum((y_test == -1) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == -1))
    
    return tp, tn, fp, fn


def precision(tp, fp):
    """Calculates the precision score, which is the proportion of positive identifications that are actually correct
    
    Args: 
        tp: true positive
        fp: false positive
        
    Returns:
        precision score
    
    """
    if(tp + fp == 0):
        return 0.0
    return tp / (tp + fp)

def recall(tp, fn):
    """Calculates the recall score, which is the proportion of positives correctly identified. 
    It is different from precision in that it does not take into account the false positives.
    
    Args:
        tp: true positive
        fn: false negative
        
    Returns:
        recall score
    """
    if(tp + fn == 0):
        return 0.0
    return tp / (tp + fn)

def f1_score(tp, fp, fn):
    """Calculates the f1 score, which is the harmonic mean of precision and recall
    
    Args:
        tp: true positive
        fp: false positive
        fn: false negative
        
    Returns:
        f1 score
    """
    denom = precision(tp, fp) + recall(tp, fn)
    if(denom == 0):
        return 0.0
    return 2 * precision(tp, fp) * recall(tp, fn) / denom

def accuracy(tp, tn, fp, fn):
    """Calculates the accuracy score, which is the proportion of correct predictions
    
    Args:
        tp: true positive
        tn: true negative
        fp: false positive
        fn: false negative
        
    Returns:
        accuracy score
    """
    denom = (tp + tn + fp + fn)
    if(denom == 0):
        return 0.0
    return (tp + tn) / denom

    