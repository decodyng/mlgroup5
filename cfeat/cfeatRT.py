__author__ = 'kensimonds'

import pandas as pd
import numpy as np
import sklearn.cross_validation as cv
from sklearn import metrics
import chooseFeature

# Dictionary of words tabulated by class (0-4)
classTabulations = open("../nBayes/classTabulationsDict.json")
classTabulationsDict = pd.io.json.read_json(classTabulations)

# Input feature set
featureSet = open("../data/kaggleFinal.json")
featureDict = pd.io.json.read_json(featureSet)
featureDict.reindex(np.random.permutation(featureDict.index))

X = featureDict
y = featureDict["rating"]

del X["rating"]

index = 0

mostCommonClassList = []
mostFrequentClassList = []

# Iterate through each review
for row in X["words"]:
    presenceScores = [i for i in range(5)]
    frequencyScores = [j for j in range(5)]
    for k in range(5):
        presenceScores[k] = []
        frequencyScores[k] = []
    mostCommonClass = 0
    mostFrequentClass = 0
    highestPres = 0.0
    highestFreq = 0.0
    # Iterate through each word in the review
    for word in row:
        # For every word compare it to the tabulated dictionary for each class
        for rating in range(5):
            try:
                # calculate presence of the word in a given class
                if np.isnan(classTabulationsDict.loc[word.lower()][rating]):
                    presenceInClass = 0
                elif classTabulationsDict.loc[word.lower()][rating] > 0:
                    presenceInClass = 1
                # calculate the frequency with which the word appears in a given class
                frequencyInClass = classTabulationsDict.loc[word.lower()][rating]/classTabulationsDict.loc["totalWords"][rating]
                if np.isnan(frequencyInClass):
                    frequencyInClass = 0

                presenceScores[rating].append(presenceInClass)
                frequencyScores[rating].append(presenceInClass)
            except KeyError:
                continue

    # Calculate average Word Presence and Word Frequency scores per review
    # Remember the class with the highest value for this review
    for rating in range(5):
        avgPres = np.mean(presenceScores[rating])
        avgFreq = np.mean(frequencyScores[rating])
        if avgPres > highestPres:
            mostCommonClass = rating
            highestProb = avgPres
        if avgFreq > highestFreq:
            mostFrequentClass = rating
            highestFreq = avgFreq

    mostCommonClassList.append(mostCommonClass)
    mostFrequentClassList.append(mostFrequentClass)
    index += 1

# Add the two new features to the dataframe
X["mostCommonClass"] = mostCommonClassList
X["mostFrequentClass"] = mostFrequentClassList

# Delete the non-numeric feature columns (chooseFeature can't handle them)
del X["review"]
del X["words"]

# Create an 85/15 train/test split
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size = 0.15)

# Create, fit, and predict the chooseFeature class
cfeat = chooseFeature.chooseFeature()
cfeat.fit(X_train, y_train)
predicted = cfeat.predict(X_test)

# Evaluate
print metrics.classification_report(y_test, predicted)
print np.mean(predicted == y_test)