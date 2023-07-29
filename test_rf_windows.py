import numpy as np
import pandas as pd

def windows(array, group_size, step_size):
    """This method takes an array and generates overlapping windows.
    ----------
    Arguments:
    array (np.array): the array to process. In this case is X_test already...
    group_size (int): the length of each window.
    step_size (int): the step used to create the windows.
    data_type (string): whether it is data or labels.

    Returns:
    groups (np.array): windowed array."""

    groups = []
    for i in range(0, array.shape[0] - group_size + 1, step_size):
        group = array[i:i + group_size]
        groups.append(group)
    
    return np.array(groups)

if __name__ == '__main__':
    
    # Read the data
    station = 901
    data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

    # Convert variable columns to np.ndarray
    X = data.iloc[:, 1:7].values
    y = data.iloc[:, -1].values
    
    X = windows(array=X, group_size=32, step_size=4)
    y = windows(array=y, group_size=32, step_size=4)
    
    # Stach up the data sets
    n_samples, n_rows, n_columns = X.shape
    X = X.reshape((n_samples * n_rows, n_columns))
    y = y.reshape((n_samples * n_rows))
    
    # Define train and test sets
    X_train, X_test, y_train, y_test = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):], y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

    # Call the model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=0)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_hat = model.predict(X_test)
    
    # Get the accuracy of the model
    from sklearn.metrics import accuracy_score, confusion_matrix
    accuracy = accuracy_score(y_test, y_hat)
    print('Accuracy', accuracy)

    # Get the number of rows labeled as anomalies in y_test
    print('Number of anomalies', len([i for i in y_test if i==1]))
    
    confusion_matrix = confusion_matrix(y_test, model.predict(X_test))
    print(confusion_matrix)
    