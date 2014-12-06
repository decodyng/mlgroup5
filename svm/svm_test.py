__author__ = 'kensimonds'
import json
import sys
sys.path.insert(0,"../kaggle/")

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import sklearn.feature_extraction.text as fe
import sklearn.cross_validation as cv
from sklearn import metrics
import makeSubmission
import pandas as pd

featureSet = open("../data/kaggleFinal.json")
featureDict = pd.io.json.read_json(featureSet)
#featureDict.reindex(np.random.permutation(featureDict.index)) shuffle


X = []
y = featureDict["rating"]

for item in featureDict["words"]:
    X.append(' '.join(item))

#rand = np.random.RandomState()
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size = 0.15) #, random_state=rand)

# print X_train

# vec = fe.CountVectorizer()
# print vec
# vec.fit(X_train)
# x = vec.transform(X_train)
# print len(vec.vocabulary_)
# print x

rt_clf = Pipeline([('vect', fe.CountVectorizer(ngram_range=(1,2))),
                   ('tfidf', fe.TfidfTransformer(norm='l2', use_idf=False, smooth_idf=True, sublinear_tf=True)),
                   ('svc', SVC(C=10.0, kernel='rbf', gamma=1.0))])

rt_clf.fit(X_train, y_train)
predicted = rt_clf.predict(X_test)

print metrics.classification_report(y_test, predicted)
print np.mean(predicted == y_test)

#makeSubmission.makeSubmission(rt_clf, "svm")

# parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#               'tfidf__norm': ('l1', 'l2', None),
#               'svc__C': (0.001, 0.01, 0.1, 1, 10, 100, 1000),
#               'svc__gamma': (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000)
#                 }
#
#
# gs_clf = GridSearchCV(rt_clf, parameters, n_jobs=-1)
# gs_clf = gs_clf.fit(X_train, y_train)
# best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, best_parameters[param_name]))
# print score


# Trial 1
# Default values produced accuracy of 0.39

# Ran a grid search
# svc__C: 0.5
# tfidf__use_idf: True
# vect__ngram_range: (1, 2)

# Using those values produced accuracy of 0.4/0.42

# Removing stop words reduced accuracy to 0.38/0.36/0.39

# Manipulated TFIDF parameters
# fe.TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
# Accuracy still not significantly different: 0.39/0.41/0.41

# Trial 1
# Default values produced accuracy of 0.39

# Ran a grid search
# svc__C: 0.5
# tfidf__use_idf: True
# vect__ngram_range: (1, 2)

# Using those values produced accuracy of 0.4/0.42

# Removing stop words reduced accuracy to 0.38/0.36/0.39

# Manipulated TFIDF parameters
# fe.TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
# Accuracy still not significantly different: 0.39/0.41/0.41

# delta tfidf:  http://ebiquity.umbc.edu/_file_directory_/papers/446.pdf
# tfidf weighing by CI:  http://www.ijcai.org/papers/0304.pdf

  precision    recall  f1-score   support

          0       0.44      0.22      0.29       166
          1       0.43      0.54      0.47       343
          2       0.28      0.18      0.22       260
          3       0.44      0.63      0.52       350
          4       0.56      0.35      0.43       184

avg / total       0.42      0.43      0.41      1303

0.425172678434

