__author__ = 'kensimonds'

def makeSubmission(clf, name):
    testInputFile = open("../data/test.tsv")
    testOutputFile = open("%s.csv" % name, "w+")
    phraseIDs = []
    inputPhrases = []
    for line in testInputFile:
        tokenized = line.replace('\n','').split('\t')
        phraseIDs.append(tokenized[0])
        inputPhrases.append(tokenized[2])

    inputPhrases = inputPhrases[1:] # remove header
    phraseIDs = phraseIDs[1:]

    predictions = clf.predict(inputPhrases)

    i = 0
    testOutputFile.write("PhraseId,Sentiment\n")
    for i in range(len(inputPhrases)):
        testOutputFile.write(phraseIDs[i] + "," + str(predictions[i]) + "\n")

    testOutputFile.close()
    testInputFile.close()