from collections import Counter
import numpy as np
import sklearn
from math import *


class chooseFeature(sklearn.base.BaseEstimator):
    """
    This defines a classifier that predicts on the basis of
      the feature that was found to have the best weighted purity, based on splitting all
      features according to their mean value. Then, for that feature split, it predicts
      a new example based on the mean value of the chosen feature, and the majority class for
      that split.
      You can of course define more variables!
    """

    def __init__(self):
        # if we haven't been trained, always return 1
        self.classForGreater= 1
        self.classForLeq = 1
        self.chosenFeature = 0
        self.featureMean = 0
        self.type = "chooseFeatureClf"

    def impurity(self, labels):
        #entropy formula:
        #- sum of P(value I) log2P(value I)

        #create counter, dictionary of labels and counts
        labelCounts = Counter(labels)
        entropy = 0
        n = float(len(labels))
        for labelValue in labelCounts:
            #calculate probability of a given label, dividing count by total N
            Pvi = labelCounts[labelValue]/n
            logPvi = log(Pvi, 2)
            sumterm = Pvi*logPvi
            entropy -= sumterm
        return entropy


    def weighted_impurity(self, list_of_label_lists):
        weighted_imp = 0
        for lst in list_of_label_lists:
            n = len(lst)
            imp = self.impurity(lst)
            #weighting: length of split times impurity of split
            weighted_imp += n*imp
        return weighted_imp


    def ftr_seln(self, data, labels):
        """return: index of feature with best weighted_impurity, when split
        according to its mean value; you are permitted to return other values as well,
        as long as the the first value is the index
        """
        #initialize arrays for impurities, means, and labels, to capture information
        #will be used by the fit function
        featureImpurities = []
        featureMeans = []
        splitLabels = []
        #loop through all features, identified by index
        for featureIndex in range(len(data[0])):
            featureCol = [lst[featureIndex] for lst in data]
            featureMean = np.mean(featureCol)
            featureMeans.append(featureMean)
            labelsGTMean = []
            labelsLTMean = []
            for j in range(len(featureCol)):
                if featureCol[j] <= featureMean:
                    labelsLTMean.append(labels[j])
                else:
                    labelsGTMean.append(labels[j])
            splitLabels.append([labelsGTMean, labelsLTMean])
            featureImpurity = self.weighted_impurity([labelsGTMean, labelsLTMean])
            featureImpurities.append(featureImpurity)
        featureImpurities = np.asarray(featureImpurities)
        bestI = np.where(featureImpurities == featureImpurities.min())
        bestI = bestI[0][0]
        print "Best index is: " + str(bestI)
        mean = featureMeans[bestI]
        splits = splitLabels[bestI]

        #returns the best index, as well as that features mean, and the splits generated by that mean
        return bestI, mean, splits



    def fit(self, data, labels):
        """
        Inputs: data: a list of X vectors
        labels: Y, a list of target values
        """
        chosenFeature, featureMean, splitLabels  = self.ftr_seln(data, labels)
        gteLabels = splitLabels[0]
        ltLabels = splitLabels[1]

        gtModeLabel = Counter(gteLabels).most_common(1)[0][0]
        lteModeLabel = Counter(ltLabels).most_common(1)[0][0]


        self.classForGreater= gtModeLabel
        self.classForLeq = lteModeLabel
        self.chosenFeature = chosenFeature
        self.featureMean = featureMean

    def predict(self, testData):
        """
        Input: testData: a list of X vectors to label.
        Check the chosen feature of each
        element of testData and make a classification decision based on it
        """
        cf = self.chosenFeature
        returnVals = []
        for data in testData:
            cfVal = data[cf]
            if cfVal <= self.featureMean:
                returnVals.append(self.classForLeq)
            else:
                returnVals.append(self.classForGreater)
        return returnVals


