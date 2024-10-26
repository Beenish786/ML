import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay

# Load the iris dataset
iris = datasets.load_iris()

# Take the first two features for visualization
X = iris.data[:, :2]
y = iris.target

# Create an SVM classifier with a linear kernel
clf = svm.SVC(kernel="linear", C=1)

# Train the classifier
clf.fit(X, y)

# Plot the decision boundaries
DecisionBoundaryDisplay.from_estimator(clf, X,
    cmap=plt.cm.Paired,response_method="predict",plot_method="pcolormesh",shading="auto",)
# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("SVM with Linear Kernel")

# Display the graph
plt.show()

