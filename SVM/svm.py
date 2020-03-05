import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

# storing the complete dataset in iris
iris = datasets.load_iris()

# we only take the first two features(sepal_length, sepal width)
X = iris.data[:, :2]

# Target variable "Species"
y = iris.target

## create a mesh-grid to show the classification regions

# defining limit for the x axis
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

# defining limit for y axis
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 0.01 step size with respect to x limit
h = (x_max / x_min)/100

# defining the meshgrid with above parameters
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

svc = SVC()
svc_fit = svc.fit(X, y)
print(svc_fit)

#plotting default svm
plt.figure(figsize = (5,5), dpi = 80)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.7)

plt.scatter(X[:, 0], X[:, 1] , c = y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Default SVM')
plt.show()

svc = SVC(kernel = 'linear')
svc.fit(X, y)

#Plotting linear svm
plt.figure(figsize = (5,5), dpi = 80)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.7)

plt.scatter(X[:, 0], X[:, 1] , c = y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Linear SVM')
plt.show()

svc = SVC(kernel = 'poly')
svc.fit(X, y)

#plotting polynomial svm
plt.figure(figsize = (5,5), dpi = 80)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1] , c = y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Default polynomial SVM')
plt.show()

#using different degree for svm(polynomial kernel)
plt.figure(figsize=(20, 5))

for i in range(2, 6):
    svc = SVC(kernel='poly', degree=i)
    svc.fit(X, y)

    plt.subplot(1, 4, i - 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('Polynomial kernel: degree = ' + str(i))

plt.show()


#The parameters that governs the performance of the 'rbf' are the parameters "C" which is the regularisation
# parameter and "GAMMA" which is the distance of margin of the classifier.

#plotting changes wrt gamma ( C = constant)
plt.figure(figsize=(15, 10))

g = [0.1, 1, 10, 15, 100, 1000]
for i in range(len(g)):
    svc = SVC(kernel='rbf', gamma=g[i])
    svc.fit(X, y)

    plt.subplot(2, 3, i + 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('rbf kernel: gamma = ' + str(g[i]))

plt.show()

#plotting changes wrt C (gamma is constant)
plt.figure(figsize=(15, 10))

c = [1, 10, 100, 1000, 10000, 100000]
for i in range(len(c)):
    svc = SVC(kernel='rbf', C=c[i], gamma=0.5)
    svc.fit(X, y)

    plt.subplot(2, 3, i + 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.7)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.title('rbf kernel: C = ' + str(c[i]))

plt.show()