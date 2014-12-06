import json
import sys
sys.path.insert(0,"../kaggle/")

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import sklearn.preprocessing as preproc
import sklearn.feature_extraction.text as fe
import sklearn.cross_validation as cv
from sklearn import metrics

# Read in the unprocessed Kaggle reviews as a JSON
featureSet = open("../data/kaggle.json")
featureDict = json.load(featureSet)

# Set X -> list of reviews and y -> list of ratings
X = [item["review"] for item in featureDict]
y = [item["rating"] for item in featureDict]

# Create the 85/15 training/test split
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size = 0.15)


# Assign Rotten Tomatoes classifier to an skLearn pipeline
rtCLF = Pipeline([('tfidf', fe.TfidfVectorizer(
                        use_idf=True, smooth_idf=True, norm='l1',
                        sublinear_tf=True, analyzer='word', max_features=10000)),
                   #('std', preproc.StandardScaler(with_mean=False)),  norm='l2', ngram_range=(1,2),
                   ('norm', preproc.Normalizer()),
                   ('svc', SVC(C=1, kernel = 'rbf', gamma=0.1, class_weight='auto'))])

# Fit the classifier and predict on test set
rtCLF.fit(X_train, y_train)
predicted = rtCLF.predict(X_test)

# Evaluate
print metrics.classification_report(y_test, predicted)
print np.mean(predicted == y_test)


# Grid Search code for exhaustive cross validation and hyperparameter tuning across all specified parameters

# parameters = {'tfidf__norm': ('l1', 'l2', None),
#               'tfidf__ngram_range': [(1,1),(1,2)],
#               'tfidf__max_features': (1000, 3000, 5000, 10000),
#               'svc__C': (0.001, 0.01, 0.1, 1, 10, 100, 1000),
#               'svc__gamma': (0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000)
#                 }
#
#
# grid = GridSearchCV(rtCLF, parameters, n_jobs=-1)
# grid = grid.fit(X_train, y_train)
# bestParameters, score, _ = max(grid.grid_scores_, key=lambda x: x[1])
# for param in parameters:
#     print param, bestParameters[param]
# print score










