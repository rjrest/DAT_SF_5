# imports
import matplotlib
matplotlib.use("AGG")
from matplotlib import pyplot as pl
import numpy as np
from sklearn import svm, datasets



# titles for the plots
titles = ['SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel',
          'LinearSVC (linear kernel)']

# set the step size in the mesh
h = .02

# set the SVM regularization parameter
C = 1.0

# set the output filename
output_file = 'svm_iris.png'


# <begin stuff for part 2>

# command line args: python file, step size, reg. param, output file
# import sys
# args = str(sys.argv)
# print args

# if sys.argv[1]:
#     h = float(sys.argv[1])
# else:
#     h = .02

# if sys.argv[2]:
#     C = float(sys.argv[2])
# else:
#     C = 1.0

# if sys.argv[3]:
#     output_file = str(sys.argv[3])
# else:
#     output_file = 'svm_iris.png'

# note: in production code you would probably want to add some type / error
# checking of the passed arguments

# notice that if we have less than 4 arguments, it is possible for one or more
# arguments to be interpreted incorrectly.  one way to avoid this problem is to
# used named arguments:

# <end stuff for part 2>


# import the iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
Y = iris.target



# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors

# we'll test four different variants of svm
svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, Y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
lin_svc = svm.LinearSVC(C=C).fit(X, Y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))



# code for plots

# loop through the 4-tuple
for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    pl.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
    pl.axis('off')

    # Plot also the training points
    pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

    pl.title(titles[i])


# save the output graphs as a .png
# pl.savefig("svm_iris.png")
pl.savefig(output_file)
