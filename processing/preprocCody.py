import json
import pandas as pd
import os
import re
import nltk

def tsvToJSON(fName):
    """
    :param fName: name of tsv file
    :return: JSONified string of tsv file
    """
    tsvData = pd.read_table(fName, header=None)
    tsvData.columns = ["review", "rating"]
    print tsvData
    outJSON = tsvData.to_json(orient='records')
    return outJSON

def splitWords(inputFName, outputFName):
    """
    :param fName: name of raw JSON file
    :return: JSON string w/ "words": [word, word, word]
    """
    inJSON = json.load(open(inputFName, "r"))
    for entry in inJSON:
        entry["words"] = entry["review"].split(" ")
    jsonOut = open(outputFName, "w")
    json.dump(inJSON, jsonOut)
    jsonOut.close()

def avgWordLength(inputFName, outputFName):
    """
    :param fName: name of JSON file w/ word splits
    :return: JSON string w/ "avgWordLength" and "numWords"
    """
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    for entry in inJSON:
        totalLength = 0
        numWords = 0
        for word in entry["words"]:
            if len(word) > 1:
                totalLength += len(word)
                numWords += 1
        entry["numWords"] = numWords
        if numWords != 0:
            entry["avgWordLength"] = round(totalLength/float(numWords), 3)
        else:
            entry["avgWordLength"] = 0
    outJSON = open(outputFName, "w")
    json.dump(inJSON, outJSON)
    outJSON.close()

def puncCount(inputFName, outputFName):
    """
    :param fName: name of JSON file w/ word splits
    :return: JSON string w/ "innerPunctuation", "exclamationPoints", "numQuestMarks"
    """
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    for entry in inJSON:
        numInnerPunc = 0
        numExclamation = 0
        numQuestion = 0
        for word in entry["words"]:
            if re.match(r'[",", ";", ".", ":"]+', word):
                numInnerPunc += 1
            if re.match(r'["?"]+', word):
                numExclamation += 1
            if re.match(r'["!"]+', word):
                numQuestion += 1
        entry["innerPunctuation"] = numInnerPunc
        entry["exclamationPoints"] = numExclamation
        entry["numQuestMarks"] = numQuestion

    outJSON = open(outputFName, "w")
    json.dump(inJSON, outJSON)
    outJSON.close()

def numFirstPerson(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    for entry in inJSON:
        numFirstPerson = 0
        for word in entry["words"]:
            if word == "I":
                numFirstPerson += 1
        entry["numFirstPerson"] = numFirstPerson

    outJSON = open(outputFName, "w")
    json.dump(inJSON, outJSON)
    outJSON.close()

def posTag(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    i = 0
    for entry in inJSON:
        numPropNoun = 0
        numOtherNoun = 0
        numPronouns = 0
        numConj = 0
        numPresVerb = 0
        numPastVerb = 0
        numParticiple = 0
        numAdj = 0
        numDet = 0
        for word in entry["words"]:
            pos = nltk.pos_tag([word])
            pos = pos[0][1]
            if re.match('NNP', pos):
                numPropNoun += 1
            elif re.match('NN.*', pos):
                numOtherNoun += 1
            elif re.match('VBD', pos):
                numPastVerb += 1
            elif re.match('VBG', pos):
                numParticiple += 1
            elif re.match('VB[Z,P]', pos):
                numPresVerb += 1
            elif re.match('[W,PR]P', pos):
                numPronouns += 1
            elif re.match('CC', pos):
                numConj += 1
            elif re.match('JJ', pos):
                numAdj += 1
            elif re.match('DT', pos):
                numDet += 1
        entry["numPropNoun"] = numPropNoun
        entry["numOtherNoun"] = numOtherNoun
        entry["numPronouns"] = numPronouns
        entry["numConj"] = numConj
        entry["numPresVerb"] = numPresVerb
        entry["numPastVerb"] = numPastVerb
        entry["numParticiple"] = numParticiple
        entry["numAdj"] = numAdj
        entry["numDet"] = numDet
        i += 1
        if i % 100 == 0:
            print i

    outJSON = open(outputFName, "w")
    json.dump(inJSON, outJSON)
    outJSON.close()

def negation(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    numProcessed = 0
    for entry in inJSON:
        modifier = None
        negativeTerritory = 0
        entry["words"]

        for j in range(len(entry["words"])):
            word = entry["words"][j]
            if word in ["not", "n't"]:
                modifier = "vrbAdj"
                negativeTerritory = 4
            elif word in ["no", "none"]:
                modifier = "nouns"
                negativeTerritory = 4
            else:
                if negativeTerritory > 0:
                    pos = nltk.pos_tag([word])
                    pos = pos[0][1]
                    if (re.match('VB[G,P,D]*', pos) or re.match('JJ', pos)) and modifier == "vrbAdj":
                        entry["words"][j] = "not_" + word
                    elif re.match('NN.*', pos) and modifier == "nouns":
                        entry["words"][j] = "not_" + word
                    negativeTerritory -= 1
        numProcessed += 1
        if numProcessed % 100 == 0:
            print numProcessed

    outJSON = open(outputFName, "w")
    json.dump(inJSON, outJSON)
    outJSON.close()

def stemming(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    porter = nltk.PorterStemmer()
    for entry in inJSON:
        for j in range(len(entry["words"])):
            word = entry["words"][j]

            if re.match('not_.*', word):
                word = word[4:]
                entry["words"][j] = "not_" + porter.stem(word)
            else:
                entry["words"][j] = porter.stem(word)
    outJSON = open(outputFName, "w")
    json.dump(inJSON, outJSON)
    outJSON.close()

def propNounConcat(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    for entry in inJSON:
        numWords = len(entry["words"])
        j = 0
        while j < numWords:
            word = entry["words"][j]
            if word[0].isupper() and j+1 < numWords:
                word2 = entry["words"][j+1]
                if word2[0].isupper():
                    joinedWord = word + "_" + word2
                    entry["words"][j] = joinedWord
                    del entry["words"][j+1]
                    numWords -= 1
            j += 1
    outJSON = open(outputFName, "w")
    json.dump(inJSON, outJSON)
    outJSON.close()



if __name__ == "__main__":
    os.chdir("..")
    if not os.path.isfile("data/kaggle.json"):
        jsonOut = open("data/kaggle.json", "w")
        print >> jsonOut, tsvToJSON("data/kaggle_roots.tsv")
        jsonOut.close()

    if not os.path.isfile("data/kaggleSplit.json"):
        splitWords("data/kaggle.json", "data/kaggleSplit.json")

    avgWordLength("data/kaggleSplit.json", "data/kaggleFinal.json")
    puncCount("data/kaggleFinal.json", "data/kaggleFinal.json")
    numFirstPerson("data/kaggleFinal.json", "data/kaggleFinal.json")
    posTag("data/kaggleFinal.json", "data/kaggleFinal.json")
    negation("data/kaggleFinal.json", "data/kaggleFinal.json")
    propNounConcat("data/kaggleFinal.json", "data/kaggleFinal.json")
    stemming("data/kaggleFinal.json", "data/kaggleFinal.json")
