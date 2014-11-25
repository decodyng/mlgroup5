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

featureSet = open("../data/kaggleNoStopWords.json")
featureDict = pd.io.json.read_json(featureSet)

X = featureDict
y = featureDict["rating"]

del X["rating"]

index = 0

mostCommonClassList = []
mostFrequentClassList = []

for row in X["words_nostopwords"]:
    presenceScores = [i for i in range(5)]
    frequencyScores = [j for j in range(5)]
    for k in range(5):
        presenceScores[k] = []
        frequencyScores[k] = []
    mostCommonClass = 0
    mostFrequentClass = 0
    highestPres = 0.0
    highestFreq = 0.0
    for word in row:
        for rating in range(5):
            # calculate presence of the word in a given class
            if np.isnan(classTabulationsDict.loc[word][rating]):
                presenceInClass = 0
            elif classTabulationsDict.loc[word][rating] > 0:
                presenceInClass = 1
            # calculate the frequency with which the word appears in a given class
            frequencyInClass = classTabulationsDict.loc[word][rating]/classTabulationsDict.loc["totalWords"][rating]
            if np.isnan(frequencyInClass):
                frequencyInClass = 0

            presenceScores[rating].append(presenceInClass)
            frequencyScores[rating].append(presenceInClass)

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

X["mostCommonClass"] = mostCommonClassList
X["mostFrequentClass"] = mostFrequentClassList

del X["review"]
del X["words"]
del X["words_nostopwords"]

print X.loc[1]

# Train and test

X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size = 0.15)

cfeat = chooseFeature.chooseFeature()
cfeat.fit(X_train, y_train)
predicted = cfeat.predict(X_test)


i = 0
correct = 0
for p in predicted:
    if p == y_test[i]:
        correct += 1

print "Correct: ", correct
print "Total: ", len(predicted)
print "Rate: ", float(correct)/len(predicted)*100