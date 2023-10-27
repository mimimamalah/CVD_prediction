import numpy as np
import implementations as imp
import metrics

    
def split_data_kfold(y, x, k):
    """Splits the data into k folds for cross validation
    
    Args:
        x: features
        y: labels
        k: number of folds
        
    Returns:
        data_splits: list of tuples containing the data splits
    """
    fold_size = x.shape[0] // k
    data_splits = []
    
    for i in range(k):
        start = i * fold_size
        end = (i+1) * fold_size
        
        x_valid = x[start:end]
        y_valid = y[start:end]
        
        x_train = np.concatenate((x[:start], x[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]))
        
        data_splits.append((x_valid, y_valid, x_train, y_train, ))
        
    return data_splits
    
    
def cross_validation_linear_regression(y, x, sgd=False, k=5,  threshold=[-0.5,-0.4], max_iters_list=[1000], gamma_list=[0.01, 0.1, 1, 10]):
    best_gamma = None
    best_max_iters = None
    best_average_f1_score = -1
    best_average_accuracy = -1
    for thresh in threshold:
        for max_iters in max_iters_list:
            for gamma in gamma_list:
                data_splits = split_data_kfold(y, x, k)
                average_f1_score = 0
                average_accuracy = 0
                
                for _, (x_valid, y_valid, x_train, y_train) in enumerate(data_splits):
                    initial_w = -np.ones((x_train.shape[1], 1))
                    
                    if(sgd):
                        w, _ = imp.mean_squared_error_sgd(y_train, x_train, initial_w, max_iters, gamma)
                    else:
                        w, _ = imp.mean_squared_error_gd(y_train, x_train, initial_w, max_iters, gamma)
                    y_pred = x_valid.dot(w)
                    y_pred_discrete = np.where(y_pred >= thresh, 1, -1)
                    
                    # Calculate evaluation metrics
                    tp, tn, fp, fn = metrics.calculate_parameters(y_valid, y_pred_discrete)
                    average_f1_score += metrics.f1_score(tp, fp, fn)
                    average_accuracy += metrics.accuracy(tp, tn, fp, fn)
                
                average_f1_score /= k # Average F1 score for the current hyperparameters
                average_accuracy /= k # Average accuracy for the current hyperparameters
                
                # Check if the current set of hyperparameters has the highest F1 score
                if average_f1_score > best_average_f1_score:
                    best_gamma = gamma
                    best_max_iters = max_iters
                    best_average_f1_score = average_f1_score
                    best_average_accuracy = average_accuracy
                    best_thresh = thresh
    
    # Print metrics for the best hyperparameters
    print(f"Best gamma value is {best_gamma}")
    print(f"Best Threshold value is {best_thresh}")
    print(f"Best max_iters value is {best_max_iters}")
    print("________________________")
    print(f"Average F1 score: {best_average_f1_score * 100:.2f} %")
    print(f"Average accuracy: {best_average_accuracy * 100:.2f} %")
    print("________________________")
    
    return best_max_iters, best_gamma   

def cross_validation_least_squares(y, x, k=5, threshold=[-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1]):
    
    data_splits = split_data_kfold(y, x, k)
    f1_sum = 0
    accuracy_sum = 0
    best_average_f1_score = -1
    best_average_accuracy = -1
    for thresh in threshold:
        average_f1_score = 0
        average_accuracy = 0
        for _, (x_valid, y_valid, x_train, y_train) in enumerate(data_splits):
            
            w, _ = imp.least_squares(y_train, x_train)
            y_pred = x_valid.dot(w)
            y_pred_discrete = np.where(y_pred >= thresh, 1, -1)    
        
            # Calculate evaluation metrics
            tp, tn, fp, fn = metrics.calculate_parameters(y_valid, y_pred_discrete)
            average_f1_score += metrics.f1_score(tp, fp, fn)
            average_accuracy += metrics.accuracy(tp, tn, fp, fn)
            
        average_f1_score /= k # Calculate the average F1 score for the current lambda value
        average_accuracy /= k # Calculate the average accuracy for the current lambda value
        
                # Check if the current lambda value has the highest F1 score
        if average_f1_score > best_average_f1_score:
            best_average_f1_score = average_f1_score
            best_average_accuracy = average_accuracy
            best_thresh = thresh
        # Print metrics for the fold with the highest F1 Score
    print(f"Best threshold value is : {best_thresh}")
    print("________________________")
    print(f"Average F1 score: {best_average_f1_score*100:.2f} %")
    print(f"Average accuracy: {best_average_accuracy*100:.2f} %")
    print("________________________")
    return best_thresh
    

        
def cross_validation_ridge_regression(y, x, k=5, threshold=[-0.5,-0.4,-0.3,-0.2], lambda_values=[0.01, 0.1, 1, 10, 100]):
    best_lambda = None
    best_average_f1_score = -1
    best_average_accuracy = -1
    
    for thresh in threshold:
        for lambda_ in lambda_values:
            data_splits = split_data_kfold(y, x, k)
            average_f1_score = 0
            average_accuracy = 0
            
            for _, (x_valid, y_valid, x_train, y_train) in enumerate(data_splits):
                w, _ = imp.ridge_regression(y_train, x_train, lambda_)
                y_pred = x_valid.dot(w)
                y_pred_discrete = np.where(y_pred >= thresh, 1, -1)    
            
                # Calculate evaluation metrics
                tp, tn, fp, fn = metrics.calculate_parameters(y_valid, y_pred_discrete)
                average_accuracy += metrics.accuracy(tp, tn, fp, fn)
                average_f1_score += metrics.f1_score(tp, fp, fn)
                
            average_f1_score /= k # Calculate the average F1 score for the current lambda value
            average_accuracy /= k # Calculate the average accuracy for the current lambda value
            
            # Check if the current lambda value has the highest F1 score
            if average_f1_score > best_average_f1_score:
                best_average_f1_score = average_f1_score
                best_average_accuracy = average_accuracy
                best_lambda = lambda_
                best_thresh = thresh
    
    # Print metrics for the fold with the highest F1 Score
    print(f"Best lambda value is {best_lambda}")
    print(f"Best threshold value is {best_thresh}")
    print("________________________")
    print(f"Average F1 score: {best_average_f1_score*100:.2f} %")
    print(f"Average accuracy: {best_average_accuracy*100:.2f} %")
    print("________________________")
    
    return best_lambda,best_thresh
                  
def cross_validation_logistic(y, x, k=5, reg=False, thresholds=[-0.5,-0.2,0], max_iters_list=[200], gamma_list=[0.01,0.1]):
    best_gamma = None
    best_max_iters = None
    best_average_f1_score = -1
    best_average_accuracy = -1


    for thresh in thresholds :
        for max_iters in max_iters_list:
            for gamma in gamma_list:
                data_splits = split_data_kfold(y, x, k)
                average_f1_score = 0
                average_accuracy = 0
                print(thresh)
                for _, (x_valid, y_valid, x_train, y_train) in enumerate(data_splits):
                    initial_w = np.random.normal(0,1,x_train.shape[1]) ## TODO TEST x_train.shape[1]
                    if(reg):
                        lambda_=1e-11
                        w, _ = imp.reg_logistic_regression(y_train, x_train, lambda_, initial_w , max_iters, gamma)
                    else:
                        w, _ = imp.logistic_regression(y_train, x_train, initial_w , max_iters, gamma)
                    y_pred = x_valid.dot(w)
                    y_pred_discrete = np.where(y_pred >= thresh, 1, -1)    
                
                    # Calculate evaluation metrics
                    tp, tn, fp, fn = metrics.calculate_parameters(y_valid, y_pred_discrete)
                    average_f1_score += metrics.f1_score(tp, fp, fn)
                    average_accuracy += metrics.accuracy(tp, tn, fp, fn)
                    
                average_f1_score /= k # Average F1 score for the current hyperparameters
                average_accuracy /= k # Average accuracy for the current hyperparameters
                
                # Check if the current set of hyperparameters has the highest F1 score
                if average_f1_score > best_average_f1_score:
                    best_gamma = gamma
                    best_max_iters = max_iters
                    best_average_f1_score = average_f1_score
                    best_average_accuracy = average_accuracy
                    best_Threshold= thresh
    
    # Print metrics for the best hyperparameters
    print(f"Best gamma value is {best_gamma}")
    print(f"Best max_iters value is {best_max_iters}")
    print(f"Best Threshold  value is {thresh}")
    print("________________________")
    print(f"Average F1 score: {best_average_f1_score * 100:.2f} %")
    print(f"Average accuracy: {best_average_accuracy * 100:.2f} %")
    print("________________________")
    
    return best_max_iters, best_gamma  
     