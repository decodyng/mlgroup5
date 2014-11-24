import os
import json

def generateTabulations(rating):
    print "Processing " + str(rating)
    inJSON = open("data/kaggleFinal.json", "r")
    records = json.load(inJSON)
    inJSON.close()
    tabulateDict = {}
    totalWords = 0
    totalDocs = 0
    for record in records:
        if record["rating"] == rating:
            for word in record["words"]:
                if tabulateDict.get(word) is None:
                    tabulateDict[word] = 0
                tabulateDict[word] += 1
                totalWords += 1
            totalDocs += 1
    tabulateDict["totalWords"] = totalWords
    tabulateDict["totalDocs"] = totalDocs
    return tabulateDict



if __name__ == "__main__":
    #for each class Y
    #generate a dictionary with the number of occurances of
    #word X in class Y
    #as well as the total number of words in class Y
    #total number of documents in class Y
    os.chdir("..")
    ratings = [0, 1, 2, 3, 4]
    allClassTabulations = {}
    for rating in ratings:
        allClassTabulations[rating] = generateTabulations(rating)

    outJSON = open("nBayes/classTabulationsDict.json", "w")
    json.dump(allClassTabulations, outJSON)
    outJSON.close()