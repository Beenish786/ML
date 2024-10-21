from sklearn.model_selection import train_test_split
from sklearn import svm , datasets
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt


iris=datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = svm.SVC(kernel='linear', C=1)


clf.fit(X,y)

clf.predict(X_test)

DecisionBoundaryDisplay.from_estimator(clf, X,
    cmap=plt.cm.Paired,response_method="predict",plot_method="pcolormesh",shading="auto",)

# Plot the data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.title("SVM with Linear Kernel")

# Display the graph
plt.show()
