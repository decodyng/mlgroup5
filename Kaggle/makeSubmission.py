__author__ = 'kensimonds'

def makeSubmission(clf, name):
    # Input: A fitted classifier
    # Output: None
    # This function creates a Kaggle submission file by making predictions and storing the resulting csv file
    # in the prescribed manner
    testInputFile = open("../data/test.tsv")  # Kaggle test set
    testOutputFile = open("%s.csv" % name, "w+") # Kaggle test submission file
    phraseIDs = []
    inputPhrases = []
    for line in testInputFile:  # Extract reviews
        tokenized = line.replace('\n','').split('\t')
        phraseIDs.append(tokenized[0])
        inputPhrases.append(tokenized[2])

    inputPhrases = inputPhrases[1:] # remove header
    phraseIDs = phraseIDs[1:]

    predictions = clf.predict(inputPhrases)  # Make predictions

    i = 0
    testOutputFile.write("PhraseId,Sentiment\n")
    for i in range(len(inputPhrases)):
        testOutputFile.write(phraseIDs[i] + "," + str(predictions[i]) + "\n")

    testOutputFile.close()
    testInputFile.close()