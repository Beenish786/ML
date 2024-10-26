import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = datasets.load_iris()

# Create a DataFrame from the iris dataset-
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Separate features (X) and target variable (y)
X = df.drop('target', axis=1)
y = df['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an SVM classifier
clf = svm.SVC(kernel='linear')

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
clf.predict(X_test)
