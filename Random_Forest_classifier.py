import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the data
data = pd.read_csv('sample_data/california_housing_train.csv')

# Separate features (X) and target variable (y)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf = RandomForestClassifier(n_estimators=10, random_state=42)

# Train the classifier
rf.fit(X_train, y_train)

# Plot one of the trees in the forest
plt.figure(figsize=(10, 8))
plot_tree(rf.estimators_[0], feature_names=X.columns, filled=True)
plt.show()
