import json
import pandas as pd
import os
import sys
from sklearn.grid_search import GridSearchCV
import re
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import KFold
from transformCSV import transform_csv, transform_sklearn_dictionary
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise, classification_report, accuracy_score, roc_auc_score
from sklearn.decomposition import RandomizedPCA

from sklearn import metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier

os.chdir("..")

def baseClassify(cosine=False):
    #26.4, 25.6, 27.8
    print "baseClassify"
    if cosine:
        print "Cosine"
    trainData = pd.read_csv("data/multinomialTrain.csv", header=0)
    dat = trainData[["rating", 'numDet', 'innerPunctuation','avgWordLength',
                          'numPresVerb',  "numFirstPerson",'numPropNoun', "numOtherNoun", "numWords", "numAdj",
                           "numPastVerb", "numConj", "exclamationPoints"]]


    if cosine:
        knn = KNeighborsClassifier(n_neighbors=21, metric=pairwise.cosine_similarity)
    else:
        knn = KNeighborsClassifier(n_neighbors=21)
    scaler = preprocessing.StandardScaler()
    scaled_knn = Pipeline([('scaler', scaler), ('knn', knn)])

    kf = KFold(len(trainData), n_folds=3, shuffle=True)
    for train, test in kf:
        trainX, trainy = transform_sklearn_dictionary(transform_csv(dat.iloc[train], target_col="rating"))
        testX, testy = transform_sklearn_dictionary(transform_csv(dat.iloc[test], target_col="rating"))
        scaled_knn.fit(trainX, trainy)
        print scaled_knn.score(testX, testy)

def allFeatureClassify(cosine=False):
    print "AllFeatureClassifier"
    if cosine:
        print "Cosine"
    trainData = pd.read_csv("data/multinomialTrain.csv", header=0)
    # dat = trainData[["rating", 'numDet', 'innerPunctuation','avgWordLength',
    #                       'numPresVerb',  "numFirstPerson",'numPropNoun', "numOtherNoun", "numWords", "numAdj",
    #                        "numPastVerb", "numConj", "exclamationPoints"]]
    dat = trainData


    if cosine:
        knn = KNeighborsClassifier(n_neighbors=21, metric=pairwise.cosine_similarity)
    else:
        knn = KNeighborsClassifier(n_neighbors=21)
    scaler = preprocessing.StandardScaler()
    scaled_knn = Pipeline([('scaler', scaler), ('knn', knn)])

    kf = KFold(len(trainData), n_folds=3, shuffle=True)
    for train, test in kf:
        trainX, trainy = transform_sklearn_dictionary(transform_csv(dat.iloc[train], target_col="rating",
                                                                    ignore_cols=["01v234", "2v34", "words",
                                                                                 "words_nostopwords", "review"]))
        testX, testy = transform_sklearn_dictionary(transform_csv(dat.iloc[test], target_col="rating",
                                                                  ignore_cols=["01v234", "2v34", "words",
                                                                                 "words_nostopwords", "review"]))
        scaled_knn.fit(trainX, trainy)
        print scaled_knn.score(testX, testy)

def useTFIDF():
    print "TFIDF"
    trainData = pd.read_csv("data/multinomialTrain.csv", header=0)
    # dat = trainData[["rating", 'numDet', 'innerPunctuation','avgWordLength',
    #                       'numPresVerb',  "numFirstPerson",'numPropNoun', "numOtherNoun", "numWords", "numAdj",
    #                        "numPastVerb", "numConj", "exclamationPoints"]]
    dat = trainData


    knn = KNeighborsClassifier(n_neighbors=21, weights='distance')
    scaler = preprocessing.StandardScaler()
    tfidf = TfidfTransformer()
    tfidf_scaled_knn = Pipeline([('tfidf', tfidf), ('knn', knn)])

    kf = KFold(len(trainData), n_folds=3, shuffle=True)
    for train, test in kf:
        trainX, trainy = transform_sklearn_dictionary(transform_csv(dat.iloc[train], target_col="rating",
                                                                    ignore_cols=["01v234", "2v34", "words","words_nostopwords",
                                                                     "review", 'numDet', 'innerPunctuation','avgWordLength','numPresVerb',  "numFirstPerson",'numPropNoun', "numOtherNoun", "numWords", "numAdj",
                                                                     "numPastVerb", "numConj", "exclamationPoints"]))
        testX, testy = transform_sklearn_dictionary(transform_csv(dat.iloc[test], target_col="rating",
                                                                  ignore_cols=["01v234", "2v34", "words","words_nostopwords",
                                                                     "review", 'numDet', 'innerPunctuation','avgWordLength','numPresVerb',  "numFirstPerson",'numPropNoun', "numOtherNoun", "numWords", "numAdj",
                                                                     "numPastVerb", "numConj", "exclamationPoints"]))
        tfidf_scaled_knn.fit(trainX, trainy)
        print tfidf_scaled_knn.score(testX, testy)
def kPCA_GS():
    trainData = pd.read_csv("data/multinomialTrain.csv", header=0)
    # dat = trainData[["rating", 'numDet', 'innerPunctuation','avgWordLength',
    #                       'numPresVerb',  "numFirstPerson",'numPropNoun', "numOtherNoun", "numWords", "numAdj",
    #                        "numPastVerb", "numConj", "exclamationPoints"]]
    print "Data read"
    dat = trainData
    n_comp = [80, 100, 120]
    ks = [100, 120, 150, 175, 200, 220]
    metrics = ['minkowski', pairwise.cosine_similarity]
    pca = RandomizedPCA()
    tfidf = TfidfTransformer()
    knn = KNeighborsClassifier()
    pipe = Pipeline([('tfidf', tfidf), ('pca', pca), ('knn', knn)])
    estimator = GridSearchCV(pipe, dict(pca__n_components=n_comp, knn__n_neighbors=ks, knn__metric=metrics))
    i = 0
    for train, test in KFold(len(trainData), n_folds=2, shuffle=True):
        print "Fold " + str(i)
        trainX, trainy = transform_sklearn_dictionary(transform_csv(dat.iloc[train], target_col="rating",
                                                                    ignore_cols=["01v234", "2v34", "words",
                                                                                 "words_nostopwords", "review"]))
        testX, testy = transform_sklearn_dictionary(transform_csv(dat.iloc[test], target_col="rating",
                                                                  ignore_cols=["01v234", "2v34", "words",
                                                                                 "words_nostopwords", "review"]))
        estimator.fit(trainX, trainy)
        print estimator.best_params_
        print estimator.best_estimator_
        fileName = "bestKNNEstimatorR2" + str(i) + ".pkl"
        with open(fileName, 'wb') as fid:
            cPickle.dump(estimator.best_estimator_, fid)

        predictions = estimator.predict(testX)
        print classification_report(testy, predictions)
        i += 1

def testPCAK(k):
    #trainData = pd.read_csv("data/multinomialRT.csv", header=0)
    trainData = pd.read_csv("data/equalClass.csv", header=0)
    trainData = trainData.fillna(value=0)
    #dat = trainData[["rating", 'numDet', 'innerPunctuation','avgWordLength',
    #                       'numPresVerb',  "numFirstPerson",'numPropNoun', "numOtherNoun", "numWords", "numAdj",
    #                        "numPastVerb", "numConj", "exclamationPoints"]]
    print "Data read"
    dat = trainData
    pca = RandomizedPCA(n_components=80)
    tfidf = TfidfTransformer()
    scaler = preprocessing.StandardScaler()
    knn = KNeighborsClassifier(n_neighbors=int(k), weights='distance')
    pipe = Pipeline([('tfidf', tfidf), ('pca', pca), ('scaler', scaler), ('knn', knn)])
    #estimator = GridSearchCV(pipe, dict(pca__n_components=n_comp, knn__n_neighbors=ks, knn__metric=metrics))
    i = 0

    # allX, allY = transform_sklearn_dictionary(transform_csv(dat, target_col="rating",
    #                                                                 ignore_cols=["words",
    #                                                                              "words_nostopwords", "review"]))


    for train, test in KFold(len(trainData), n_folds=3, shuffle=True):
        print "Fold " + str(i)
        trainX, trainy = transform_sklearn_dictionary(transform_csv(dat.iloc[train], target_col="rating",
                                                                     ignore_cols=["words",
                                                                                  "words_nostopwords", "review"]))
        testX, testy = transform_sklearn_dictionary(transform_csv(dat.iloc[test], target_col="rating",
                                                                   ignore_cols=["words",
                                                                                  "words_nostopwords", "review"]))
        #trainX, trainy = transform_sklearn_dictionary(transform_csv(dat.iloc[train], target_col="rating"))
        #testX, testy = transform_sklearn_dictionary(transform_csv(dat.iloc[test], target_col="rating"))
        #estimator.fit(trainX, trainy)
        pipe.fit(trainX, trainy)
        #print estimator.best_params_
        #print estimator.best_estimator_
        #fileName = "bestKNNEstimatorR2" + str(i) + ".pkl"
        #with open(fileName, 'wb') as fid:
        #    cPickle.dump(estimator.best_estimator_, fid)

        predictions = pipe.predict(testX)
        print classification_report(testy, predictions)
        print accuracy_score(testy, predictions)
        i += 1

def testPCAKFull(k, train=True, i=0):

    #dat = trainData[["rating", 'numDet', 'innerPunctuation','avgWordLength',
    #                       'numPresVerb',  "numFirstPerson",'numPropNoun', "numOtherNoun", "numWords", "numAdj",
    #                        "numPastVerb", "numConj", "exclamationPoints"]]
    print "Data read"

    pca = RandomizedPCA(n_components=80)
    tfidf = TfidfTransformer()
    scaler = preprocessing.StandardScaler()
    knn = KNeighborsClassifier(n_neighbors=int(k), weights='distance')
    pipe = Pipeline([('tfidf', tfidf), ('pca', pca), ('scaler', scaler), ('knn', knn)])
    #estimator = GridSearchCV(pipe, dict(pca__n_components=n_comp, knn__n_neighbors=ks, knn__metric=metrics))


    if train:
        trainData = pd.read_csv("data/equalClass.csv", header=0)
        trainData = trainData.fillna(value=0)
        dat = trainData
        #dat = trainData[["rating", 'numDet', 'innerPunctuation','avgWordLength',
        #                  'numPresVerb',  "numFirstPerson",'numPropNoun', "numOtherNoun", "numWords", "numAdj",
        #                "numPastVerb", "numConj", "exclamationPoints"]]
        allX, allY = transform_sklearn_dictionary(transform_csv(dat, target_col="rating",
                                                                     ignore_cols=["words", "01v234", "2v34", "words_nostopwords", "review"]))
        #allX, allY = transform_sklearn_dictionary(transform_csv(dat, target_col="rating"))
        pipe.fit(allX, allY)
        with open("knn/equalKNNFull.pkl", 'wb') as knnPCK:
                cPickle.dump(pipe, knnPCK)
    else:
        filename = "data/multinomialTest.csv"
        testData = pd.read_csv(filename, header=0)
        #testData = testData.fillna(value=0)
        dat = testData
        testX, realY = transform_sklearn_dictionary(transform_csv(dat, target_col="rating",
                                                                     ignore_cols=["words_nostopwords", "01v234", "2v34", "words", "review"]))
        #testX, ids = transform_sklearn_dictionary(transform_csv(dat, target_col="phraseID"))
        with open("knn/equalKNNFull.pkl", 'r') as knnPCK:
            knnLoaded = cPickle.load(knnPCK)

        predictions = knnLoaded.predict(testX)
        print classification_report(realY, predictions)
        print accuracy_score(realY, predictions)
        # outDF = pd.concat([ids, predictions], axis=1)
        # outName = "nBayes/kagglePredictions_HF" + str(i) + ".csv"
        # outDF.to_csv(outName)

if __name__ == "__main__":

    #allFeatureClassify()
    #baseClassify(cosine=True)
    #useTFIDF()
    #allFeatureClassify(cosine=True)
    #kPCA_GS()
    testPCAK(sys.argv[1])
    #testPCAKFull(sys.argv[1], train=True)
    #testPCAKFull(sys.argv[1], train=False)

