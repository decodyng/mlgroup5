__author__ = 'kensimonds'

from nltk.corpus import stopwords
import json

def removeStopWords(inputFName, outputFName):
    '''
    :param inputFName: name of JSON file with word splits
    :param outputFName: name of output JSON file
    :return: JSON file with stop words removed
    '''
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    stop = stopwords.words('english')
    for entry in inJSON:
        entry["words_nostopwords"] = []
        for word in entry["words"]:
            if word not in stop:
                entry["words_nostopwords"].append(word)
    outJSON = open(outputFName, "w")
    json.dump(inJSON, outJSON)
    outJSON.close()


removeStopWords("../data/kaggleFinal.json", "../data/kaggleNoStopWords.json")