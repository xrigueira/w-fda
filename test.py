import numpy as np
import pandas as pd

# Read the data
station = 901
data = pd.read_csv(f'data/labeled_{station}_pro.csv', sep=',', encoding='utf-8')

# Convert variable columns to np.ndarray
X = data.iloc[:, 1:7].values
y = data.iloc[:, -1].values

# Split into train and test sets
X_train, X_test, y_train, y_test = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):], y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

num_rows_train = X_train.shape[0]
num_rows_test = X_test.shape[0]

# Save the original order
original_indices_train = np.arange(num_rows_train)
original_indices_test = np.arange(num_rows_test)

# Generate random indices to shuffle the data
random_indices_train = np.random.permutation(num_rows_train)
random_indices_test = np.random.permutation(num_rows_test)

# Use the random indices to shuffle the data
shuffled_X_train = X_train[random_indices_train]
shuffled_y_train = y_train[random_indices_train]
shuffled_X_test = X_test[random_indices_test]
shuffled_y_test = y_test[random_indices_test]

# Use the original indices to return the data to its original order
restored_X_train = X_train[original_indices_train]
restored_y_train = y_train[original_indices_train]
restored_X_test = X_test[original_indices_test]
restored_y_test = y_test[original_indices_test]

# data_test = pd.DataFrame()
# data_test['Date'] = data.date[:int(len(X) * 0.8)]
# data_test['ammonium'] = data.ammonium_901[:int(len(X) * 0.8)]
# data_test['shuffled_amm'] = shuffled_X_train[:, 0]
# data_test['restored_amm'] = restored_X_train[:, 0]
# data_test['label'] = data.label[:int(len(X) * 0.8)]
# data_test['shuffled_label'] = shuffled_y_train
# data_test['restored_label'] = restored_y_train

# data_test.to_csv('data_test.csv')

# Implement Random Forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=0)

# Fit the model to the training data
model.fit(shuffled_X_train, shuffled_y_train)

# Make predictions on the testing data
y_hat = model.predict(shuffled_X_test)

# # Get the accuracy of the model
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(shuffled_y_test, y_hat)
print('Accuracy', accuracy)

# Get the number of rows labeled as anomalies in y_test
print('Number of anomalies', len([i for i in shuffled_y_test if i==1]))

# Display the confusion matrix
confusion_matrix = confusion_matrix(shuffled_y_test, model.predict(shuffled_X_test))

print(confusion_matrix)

# I may need to shuffled X and y and not their train and test sets