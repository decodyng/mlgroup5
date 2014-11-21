__author__ = 'cindi'
import argparse
import sys
import pandas as pd
import zeroR
from sklearn import cross_validation
import numpy as np
from sklearn.utils import shuffle
import firstFeatureClassifier
import chooseFeature
from sklearn import datasets


def transform_sklearn_dictionary(input_dict):
    """ Input: input_dict: a Python dictionary or dictionary-like object containing
    at least information to populate a labeled dataset, L={X,y}
    return:
    X: a list of lists. The length of inner lists should be the number of features,
     and the length of the outer list should be the number of examples.
    y: a list of target variables, whose length is the number of examples.
    X & y are not required to be numpy arrays, but you may find it convenient to make them so.
    """
    # TODO: Your code here
    X = input_dict["data"]
    y = np.ravel(np.asarray(input_dict["target"]))
    return X, y



def transform_csv(data, target_col=0, ignore_cols=None):
    """ Input: data: a pandas DataFrame
    return: a Python dictionary with same keys as those used in sklearn's iris dataset
    (you don't have to create an object of the same data type as those in sklearn's datasets)
    """
    my_dictionary = {}
    my_dictionary["target"] = data.ix[:,[target_col]]

    #check for numeric index on target
    try:
        target_col = float(target_col)
        xdata = data.drop(data.columns[target_col], axis=1)
    except ValueError:
        xdata = data.drop(target_col, axis=1)

    #check for numeric index on ignore columns
    try:
        ig = ignore_cols[0]
        try:
            float(ig)
            xdata = xdata.drop(data.columns[ignore_cols], axis=1)
        except ValueError:
            xdata = xdata.drop(ignore_cols, axis=1)
    except TypeError:
        pass


    my_dictionary["data"] = [list(x) for x in xdata.itertuples()]
    my_dictionary["feature_names"] = xdata.columns.values
    my_dictionary["target_names"] = "target"
    my_dictionary["DESCR"] = "Hi there!"
    return my_dictionary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name", help="must be either iris, digits, nursery or myst", type=str)
    parser.add_argument("-m", help="Specifies model to apply, default is zeroR", type=str)

    args = parser.parse_args()
    task_names = ['iris', 'digits', 'myst', 'nursery']

    X, y = None, None
    # initialize dataset depending on input
    if args.task_name == 'iris':
        X, y = transform_sklearn_dictionary(datasets.load_iris())
    elif args.task_name == 'digits':
        X, y = transform_sklearn_dictionary(datasets.load_digits())
    elif args.task_name == 'myst':
        rawdata = pd.read_csv('mystery.csv', header=0)
        mystery_dictionary = transform_csv(rawdata)
        X, y = transform_sklearn_dictionary(mystery_dictionary)
    elif args.task_name == 'nursery':
        rawdata = pd.read_csv('nursery.csv', header=0)
        mystery_dictionary = transform_csv(rawdata, target_col="target")
        X, y = transform_sklearn_dictionary(mystery_dictionary)
    else:
        print "illegal dataset %s. Options are: %s" % (args.task_name, task_names)


    models = ['zeroR', 'firstFeature', 'chooseFeature']
    model = 'cf'
    if args.m:
        if args.m == 'firstFeature':
            model = 'ff'
        elif args.m == 'chooseFeature':
            model == 'cf'
        elif args.m != 'zeroR':
            print "illegal model type %s. Options are: %s" % (args.m, models)
            sys.exit()

    if model == 'zeroR':
        clf = zeroR.zeroR()
    elif model == 'cf':
        clf = chooseFeature.chooseFeature()
    else:
        clf = firstFeatureClassifier.firstFeatureClassifier()
    clf.fit(X, y)
    print X[1]
    print y[1]
    print clf.predict(X[:1])
    X, y = shuffle(X, y)
    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print scores


