__author__ = 'tunc'

from nltk.tokenize import RegexpTokenizer
import json

from nltk.tokenize import RegexpTokenizer
import json

def split_clean_words(inputFName,outputFName):
    '''
    splits the sentences. Capitalizes words. 
    Removes everything except [A-Z]
    '''

    tokenizer = RegexpTokenizer(r'([A-Z]|[a-z])+')
    inJSON = json.load(open(inputFName, "r"))

    for entry in inJSON:
        entry["words_cleaned"] = [s.upper() for s in tokenizer.tokenize(entry["review"])]
    jsonOut = open(outputFName, "w")
    json.dump(inJSON, jsonOut)
    jsonOut.close()
    
def set_of_AllWords(inputFName,outputFName):
    inJSON = json.load(open(inputFName, "r"))
    s=set([])
    for i in range(len(inJSON)):
        t=set(inJSON[i]['words_cleaned'])
        s=s.union(t)
        
    with open(outputFName, 'wb') as outfile:
        json.dump(list(s), outfile)

        
        
if __name__ == "__main__":
    split_clean_words("../data/kaggle.json", "../data/kaggle_cleaned_words.json")
    set_of_AllWords("../data/kaggle_cleaned_words.json","../data/all_words.json")
