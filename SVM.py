from sklearn import svm #spot vector machine
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the data from a CSV file (replace 'your_data.csv' with your file)
data = pd.read_csv('sample_data/california_housing_train.csv')

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1]  # Select all columns except the last one
y = data.iloc[:, -1]   # Select the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
clf = svm.SVC()

# Train the classifier
clf.fit(X_train, y_train)


# Predict on the test set
clf.predict(X_test)
