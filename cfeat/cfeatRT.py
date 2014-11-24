__author__ = 'kensimonds'

import pandas as pd
import numpy as np
import nltk
import json
import sklearn.cross_validation as cv
import chooseFeature

# p19-22 TFIDF, Freq Dist, Bigrams, Collocation

# data = pd.read_table("../data/kaggle_roots.tsv", header = None)
#
# words = data[0]
# ratings = data[1]

classTabulations = open("../nBayes/classTabulationsDict.json")
classTabulationsDict = pd.io.json.read_json(classTabulations)

featureSet = open("../data/kaggleFinal.json")
featureDict = pd.io.json.read_json(featureSet)

X = featureDict
y = featureDict["rating"]

del X["rating"]

presenceScores = [0, 1, 2, 3, 4]
presenceScores[0] = []
presenceScores[1] = []
presenceScores[2] = []
presenceScores[3] = []
presenceScores[4] = []


for row in X["words"]:
    mostCommonClass = 0
    highestProb = 0.0
    for word in row:
        for rating in range(5):
            presenceInClass = classTabulationsDict.loc[word][rating]/classTabulationsDict.loc["totalWords"][rating]
            if np.isnan(presenceInClass):
                presenceInClass = 0
            presenceScores[rating].append(presenceInClass)
    for rating in range(5):
        avgProb = np.mean(presenceScores[rating])
        if avgProb > highestProb:
            mostCommonClass = rating
            highestProb = avgProb
    print mostCommonClass


#print classTabulationsDict

# X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size = 0.2)
#
# cfeat = chooseFeature.chooseFeature()
# cfeat.fit(X_train, y_train)


#chooseFeature.predict