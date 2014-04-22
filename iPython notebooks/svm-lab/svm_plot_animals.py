# lab 13 - python code

# notes:
# - remember that in python indentation is used to indicate code blocks
# - do not mix tabs and spaces when indenting (4 spaces is standard)
# - to check the import path you can use sys.path

# refs:
# - code adapted from http://www.yaksis.com/posts/why-use-svm.html


# imports
import numpy as np
import pylab as pl
import pandas as pd
from sklearn import svm
from sklearn import linear_model
from sklearn import tree


# sample variable (for showing import functionality)
# version = '0.1.0'

# sample function (for showing import functionality)
def add_vals(a, b=0):
    """ adds two values a,b and returns the result

    Note:
        the second argument has a default value of 0
    """
    return a + b



# dictionary of classifiers (key is a string, value is a class)
clfs = {
    "SVM": svm.SVC(),
    "Logistic" : linear_model.LogisticRegression(),
    "Decision Tree": tree.DecisionTreeClassifier(),
}



# used to read in our animal locations
def survey_animals(filename):
    """ Survey the animals

    This is a doc string - notice a few things:

        a. uses the triple " syntax
        b. located at the very top of the function

    Args:
        filename: name of the file to open

    Returns:
        df: a pandas dataframe with animal data

    """

    data = open(filename).read()
    # data = [row.split('\t') for row in data.strip().split('\n')]
    data = [row.split(',') for row in data.strip().split('\n')]

    # print data

    animals = []
    for y, row in enumerate(data):
        # print y
        for x, item in enumerate(row):
            # x's are sheep, o's are wolves
            if item in ['o', 'x']:
                animals.append([x, y, item])
                # print animals

    df = pd.DataFrame(animals, columns=["x", "y", "animal"])
    df['animal_type'] = df.animal.apply(lambda x: 0 if x=="x" else 1)

    return df



def plot_results(clf, clf_name, df, plt_nmbr, plot=False, lgd=False):
    """ Plot the results of a classifier

    Args:
        clf: classifier (classifier object)
        clf_name: classifier name (string)
        df: pandas dataframe containing data
        plt_nmbr: plot number (integer)

    Returns:
        None

    Notes:
        This function also manipulates a global plt instance.

    """

    x_min, x_max = df.x.min() - .5, df.x.max() + .5
    y_min, y_max = df.y.min() - .5, df.y.max() + .5

    # step between points. i.e. [0, 0.02, 0.04, ...]
    step = .02
    # to plot the boundary, we're going to create a matrix of every possible point
    # then label each point as a wolf or cow using our classifier
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # this gets our predictions back into a matrix
    Z = Z.reshape(xx.shape)

    # create a subplot (we're going to have more than 1 plot on a given image)
    pl.subplot(2, 2, plt_nmbr)
    # plot the boundaries
    pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)

    # plot the wolves and cows
    for animal in df.animal.unique():
        pl.scatter(df[df.animal==animal].x,
                   df[df.animal==animal].y,
                   marker=animal,
                   label="cows" if animal=="x" else "wolves",
                   color='black',
                   c=df.animal_type, cmap=pl.cm.Paired)
    pl.title(clf_name)
    if lgd:
        pl.legend(loc="best")

    # if the plot argument is true then plot immediately, otherwise, just
    # keep building combined plot.
    if plot:
        pl.show()

    return None



def plot_all(clfs, data_file, output_file='out.png'):
    """

    Args:
        clfs:
        data_file:
        output_file:

    Returns:

    Notes:
        - There is a problem with generating pdfs with this current code!  Use png.

    """

    # load the data file into dataframe
    df = survey_animals(data_file)

    # fit and plot
    plt_nmbr = 1
    train_cols = ["x", "y"]

    # loop through our classifiers dictionary
    for clf_name, clf in clfs.iteritems():
        clf.fit(df[train_cols], df.animal_type)
        plot_results(clf, clf_name, df, plt_nmbr)
        plt_nmbr += 1

    # pl.show()
    pl.savefig(output_file)


# The expression below is a common idiom that is often used to trigger
# different behavior based on how the file is accessed:
#
# 1.  execute the file (e.g. from the command line):     python lab13.py
# 2.  import the file (interpreter or other file):       import lab13
#
# When the file is executed (#1) above, the built-in attribute __name__ will be
# set to the string "__main__", so the expression below will evaluate to 'true'
# and the corresponding code block will be run.  Otherwise, any named objects
# defined in the file, such as variables and functions, will be added to the
# namespace but will not be executed.

if __name__ == "__main__":

    # print 'lab13 - this file was executed!'
    pass

else:

    # print 'lab13 - this file was imported!'
    pass
