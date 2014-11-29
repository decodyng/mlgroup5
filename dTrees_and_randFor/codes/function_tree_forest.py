import json
import numpy as np
import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
import random
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier

#Functions: Data Transform

def transform_sklearn_dictionary(input_dict):
    """ Input: input_dict: a Python dictionary or dictionary-like object containing
    at least information to populate a labeled dataset, L={X,y}
    return:
    X: a list of lists. The length of inner lists should be the number of features,
     and the length of the outer list should be the number of examples.
    y: a list of target variables, whose length is the number of examples.
    X & y are not required to be numpy arrays, but you may find it convenient to make them so.
    """
    X=np.asarray(input_dict['data'])
    y=np.asarray(input_dict['target'])
    return X, y


def transform_csv(data, target_col=0, ignore_cols=None):
    """ Input: data: a pandas DataFrame
    return: a Python dictionary with same keys as those used in sklearn's iris dataset
    (you don't have to create an object of the same data type as those in sklearn's datasets)
    """
    if target_col==0:target_col="target"
    if ignore_cols==None: ignore_cols=[]

    df_feature_names=[i for i in data.columns.values if not (i==target_col or i in ignore_cols)]
    df_data=data[df_feature_names]
    df_target=data[target_col]
    df_target_names=df_target.unique()

    my_dictionary={}
    my_dictionary['feature_names']=np.asarray(df_feature_names)
    my_dictionary['data']=np.asarray(df_data)
    my_dictionary["target"]=np.asarray(df_target)
    my_dictionary['target_names']=np.asarray(df_target_names)
    my_dictionary['DESCR']=""
    return my_dictionary


# Functions: Prepare Data Files


def split_train_test(inputFName,outputFName,train_path,test_path,test_percentage=0.15,shuffle=False):
    inJSON = json.load(open(inputFName, "r"))
    if shuffle:
        random.shuffle(inJSON)
        
        
    test_len=int(len(inJSON)*test_percentage)
    train_len=int(len(inJSON)-test_len)

    
    jsonOut = open(test_path+outputFName, "w")
    json.dump(inJSON[:test_len], jsonOut)
    jsonOut.close()
    jsonOut = open(train_path+outputFName, "w")
    json.dump(inJSON[test_len:], jsonOut)
    jsonOut.close()
    

    
def lower_words(inputFName,outputFName):
    '''
    removes punctuation from at the end and beginning of words
    split_clean_words
    '''
    inJSON = json.load(open(inputFName, "r"))
    my_lower_words=[]
    for entry in inJSON:
        my_lower_words.append({'words':[elt.lower() for elt in entry['words']]})
        
    with open(outputFName, 'wb') as outfile:
        json.dump(my_lower_words, outfile)

    
def remove_punctuation(inputFName,outputFName):
    '''
    removes punctuation from at the end and beginning of words
    split_clean_words
    '''
    inJSON = json.load(open(inputFName, "r"))
    my_remove_punctuation=[]
    for entry in inJSON:
        dummy={}
        exclude = "?:!.,;.-_`~+=#()/\|][*' "
        dummy['words']=entry['words']
        dummy['words']=[i.strip(exclude) for i in entry['words'] if i.strip(exclude) !=[] ]
        dummy2={}
        dummy2['words']=[]
        # remove only-exclude or entries
        for s in dummy['words']:
            s = ''.join(ch for ch in s if ch not in exclude)
            if not (s=='' or s=='s'):
                dummy2['words'].append(s)
        my_remove_punctuation.append(dummy2)
        
    with open(outputFName, 'wb') as outfile:
        json.dump(my_remove_punctuation, outfile)
    
def set_of_AllWords(inputFName,outputFName):
    inJSON = json.load(open(inputFName, "r"))
    s=set([])
    for i in range(len(inJSON)):
        t=set(inJSON[i]['words'])
        s=s.union(t)
        
    with open(outputFName, 'wb') as outfile:
        json.dump(list(s), outfile)


def splitWords(inputFName, outputFName):
    """
    :param fName: name of raw JSON file
    :return: JSON string w/ "words": [word, word, word]
    """
    inJSON = json.load(open(inputFName, "r"))
    split_Words=[]
    for entry in inJSON:
        entry["words"] = entry["review"].split(" ")
        split_Words.append( {'words':entry["words"]})
    jsonOut = open(outputFName, "w")
    json.dump(split_Words, jsonOut)
    jsonOut.close()  

def avgWordLength(inputFName, outputFName):
    """
    :param fName: name of JSON file w/ word splits
    :return: JSON string w/ "avgWordLength" and "numWords"
    """
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    avgWord=[]
    for entry in inJSON:
        dummy={}
        totalLength = 0
        numWords = 0
        for word in entry["words"]:
            if len(word) > 1:
                totalLength += len(word)
                numWords += 1
        dummy["numWords"] = numWords
        if numWords != 0:
            dummy["avgWordLength"] = round(totalLength/float(numWords), 3)
        else:
            dummy["avgWordLength"] = 0
        avgWord.append(dummy)
    outJSON = open(outputFName, "w")
    json.dump(avgWord, outJSON)
    outJSON.close()
def puncCount(inputFName, outputFName):
    """
    :param fName: name of JSON file w/ word splits
    :return: JSON string w/ "innerPunctuation", "exclamationPoints", "numQuestMarks"
    """
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    punc_Count=[]
    for entry in inJSON:
        dummy={}
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
        dummy["innerPunctuation"] = numInnerPunc
        dummy["exclamationPoints"] = numExclamation
        dummy["numQuestMarks"] = numQuestion
        punc_Count.append(dummy)

    outJSON = open(outputFName, "w")
    json.dump(punc_Count, outJSON)
    outJSON.close()

def isFirstPerson(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    num_FirstPerson=[]
    for entry in inJSON:
        dummy={}
        numFirstPerson = 0
        for word in entry["words"]:
            if word == "I":
                numFirstPerson = 1
        dummy["isFirstPerson"] = numFirstPerson
        num_FirstPerson.append(dummy)

    outJSON = open(outputFName, "w")
    json.dump(num_FirstPerson, outJSON)
    outJSON.close()
def posTag(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    i = 0
    pos_Tag=[]
    for entry in inJSON:
        dummy={}
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
        dummy["numPropNoun"] = numPropNoun
        dummy["numOtherNoun"] = numOtherNoun
        dummy["numPronouns"] = numPronouns
        dummy["numConj"] = numConj
        dummy["numPresVerb"] = numPresVerb
        dummy["numPastVerb"] = numPastVerb
        dummy["numParticiple"] = numParticiple
        dummy["numAdj"] = numAdj
        dummy["numDet"] = numDet
        pos_Tag.append(dummy)
        i += 1
        if i % 1000 == 0:
            print i

    outJSON = open(outputFName, "w")
    json.dump(pos_Tag, outJSON)
    outJSON.close()

def negation(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    numProcessed = 0
    stop = stopwords.words('english')
    mynegation=[]
    for entry in inJSON:
        modifier = None
        negativeTerritory = 0

        for j in range(len(entry["words"])):
            word = entry["words"][j]
            if word in ["not", "n't","hardly"]:
                modifier = "vrbAdj"
                negativeTerritory = 4
            elif word in ["no", "none"]:
                modifier = "nouns"
                negativeTerritory = 4
            else:
                if negativeTerritory > 0:
                    pos = nltk.pos_tag([word])
                    pos = pos[0][1]
                    if ((re.match('VB[G,P,D]*', pos) or re.match(('JJ|RB'), pos)) and modifier == "vrbAdj"):
                        if word not in stop: entry["words"][j] = "not_" + word
                    elif (re.match('NN.*', pos) and modifier == "nouns"):
                        if word not in stop: entry["words"][j] = "not_" + word
                    negativeTerritory -= 1
        mynegation.append({'words':entry["words"]})
        numProcessed += 1
        if numProcessed % 1000 == 0:
            print numProcessed

    outJSON = open(outputFName, "w")
    json.dump(mynegation, outJSON)
    outJSON.close()

def stemming(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    porter = nltk.PorterStemmer()
    my_stemming=[]
    for entry in inJSON:
        dummy={}
        for j in range(len(entry["words"])):
            word = entry["words"][j]

            if re.match('not_.*', word):
                word = word[4:]
                entry["words"][j] = "not_" + porter.stem(word)
            else:
                entry["words"][j] = porter.stem(word)
        dummy['words']=entry['words']
        my_stemming.append(dummy)
    outJSON = open(outputFName, "w")
    json.dump(my_stemming, outJSON)
    outJSON.close()

def propNounConcat(inputFName, outputFName):
    inFile = open(inputFName, "r")
    inJSON = json.load(inFile)
    inFile.close()
    my_propNounConcat=[]
    for entry in inJSON:
        dummy={}
        numWords = len(entry["words"])
        j = 0
        while j < numWords:
            word = entry["words"][j]
            if word[0].isupper() and j+1 < numWords:
                word2 = entry["words"][j+1]
                if word2[0].isupper() and j>0:
                    joinedWord = word + "_" + word2
                    entry["words"][j] = joinedWord
                    del entry["words"][j+1]
                    numWords -= 1
            j += 1
        dummy['words']=entry['words']
        my_propNounConcat.append(dummy)
    outJSON = open(outputFName, "w")
    json.dump(my_propNounConcat, outJSON)
    outJSON.close()

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
    stop.append('nt')
    my_removeStopWords=[]
    for entry in inJSON:
        dummy={}
        entry["words_nostopwords"] = []
        for word in entry["words"]:
            if word not in stop:
                entry["words_nostopwords"].append(word)
        dummy['words']=entry["words_nostopwords"]
        my_removeStopWords.append(dummy)
    outJSON = open(outputFName, "w")
    json.dump(my_removeStopWords, outJSON)
    outJSON.close()

# Functions: Prepare Data Frames I

def prepare_data(current_path,inputFilenames,outputFilenames,myFunctions):
    for i in range(len(myFunctions)):
        
        myFunction=myFunctions[i]      
        inputFName=current_path+inputFilenames[i]
        outputFName=current_path+outputFilenames[i]
        myFunction(inputFName, outputFName)

def addFrequencies(inputFName1,inputFName2,myindex):

    inJSON1 = json.load(open(inputFName1, "r"))
    inJSON2 = json.load(open(inputFName2, "r"))
    
    word_category=pd.DataFrame(inJSON1)
    word_category.columns=['words']
    if not 'rating' in pd.DataFrame(inJSON2).columns:
        # add a rating column to prevent the code to fail for the test set in Kaggle
        word_category['rating']=np.array([0]*word_category.shape[0])
    else:  
        word_category['rating']=pd.DataFrame(inJSON2)['rating']
    if not myindex==None:
        word_category=word_category.ix[myindex,:]
        

    ratings_words={}
    for i in range(0,5):
        ratings_words[i]=word_category[word_category['rating']==i][['words']]

    ratings_freq={}
    for i in range(0,5):
        a=list(ratings_words[i].ix[:,0])
        aa=np.concatenate(a)
        ratings_freq[i]=nltk.FreqDist(aa)
        min_len=len(ratings_freq[0])
        if len(ratings_freq[i])<min_len:
            min_len=len(ratings_freq[i])


    def fregFunc(elt,k):
        return sum([ratings_freq[k][i] if  ratings_freq[k].has_key(i) else 0 for i in elt])

    for i in range(0,5):
        dummy=word_category['words'].map(lambda x: fregFunc(x,i))
        dummy=(dummy-np.mean(dummy))/np.std(dummy)
        word_category['freq'+str(i)]=dummy
        
    return (word_category,ratings_words)

def addTFIDF(inputFName1,inputFName2,myindex):
    word_category,ratings_words=addFrequencies(inputFName1,inputFName2,myindex)
    #print(word_category)
    ratings_text={}
    #create text files for each category
    for i in range(5):
        ratings_text[i]=list(ratings_words[i].ix[:,0])
        ratings_text[i]=np.concatenate(ratings_text[i])
        ratings_text[i]=' '.join(ratings_text[i])
    ratings_text=list(ratings_text.values())
    tfidf = TfidfVectorizer()
    tfs = tfidf.fit_transform(ratings_text)
    ratings_tfidf={}
    for i in range(5):  
        response = tfidf.transform([ratings_text[i]])
        feature_names = tfidf.get_feature_names()
        ratings_tfidf[i]={}
        for col in response.nonzero()[1]:
            ratings_tfidf[i][feature_names[col]]=response[0, col]
    def tfidfFunc(elt,k):
        return sum([ratings_tfidf[k][i] if  ratings_tfidf[k].has_key(i) else 0 for i in elt])

    for i in range(0,5):
        dummy=word_category['words'].map(lambda x: tfidfFunc(x,i))
        word_category['tfidf'+str(i)]=dummy
    return word_category,ratings_words

def addFeatures(df,inputFiles,current_path,myindex):
    for myfile in inputFiles:
        inputFName=current_path+myfile
        inJSON = json.load(open(inputFName, "r"))
        dummy=pd.DataFrame(inJSON)
        cols=[i for i in dummy.columns if not i=='words']
        if myindex==None:
            df[cols]=dummy
        else:
            df[cols]=dummy.ix[myindex,:]

def addWords(word_category,ratings_words,numWords,myindex,all_words=[]):
    ratings_set={}
    if  all_words==[]:
        for i in range(0,5):
            a=list(ratings_words[i].ix[:,0])
            aa=np.concatenate(a)
            ratings_set[i]=set(aa)
            min_len=len(ratings_set[0])
            if len(ratings_set[i])<min_len:
                min_len=len(ratings_set[i])

        # select most popular words, and equal numbers from all classes
        print('start word selection')
        for i in range(0,5):
            good_set= nltk.FreqDist(ratings_set[i]).most_common(min(numWords,min_len))
            good_set=zip(*good_set)[0]
            ratings_set[i]=ratings_set[i].intersection(good_set)
        print('stop word selection')


        ratings_intersect=ratings_set[i]
        for i in range(0,5):
            ratings_intersect=ratings_intersect.intersection(ratings_set[i])

        ratings_union=set([])
        for i in range(0,5):
            ratings_union=ratings_union.union(ratings_set[i])

        ratings_diff=ratings_union.difference(ratings_intersect)
        for s in ratings_diff:
            is_crap=1
            for ss in s:
                is_crap=is_crap*(not ss.isalpha())
            if is_crap==0:
                all_words.append(s)
    def is_words_in(word, elt):
        return (word in elt)*1

    words_df=pd.DataFrame()
    words_df['rating_']=word_category['rating']
    for word in all_words:
        words_df[word]=word_category['words'].map(lambda x: is_words_in(word,x) )
    return {'words_df':words_df,'all_words':all_words}

#Functions: Prepare Data Frames II

def create_Train_Test_files(my_paths,my_path_keys):


    inputFilenames=["kaggle.json","splitWords.json","splitWords.json","remove_punctuation.json",\
                    "remove_punctuation.json","remove_punctuation.json","remove_punctuation.json",\
                    "negation.json","propNounConcat.json","lower_words.json","removeStopWords.json"]
    outputFilenames=["splitWords.json","puncCount.json","remove_punctuation.json","avgWordLength.json",\
                     "isFirstPerson.json","posTag.json","negation.json","propNounConcat.json","lower_words.json",\
                     "removeStopWords.json","stemming.json"]
    myFunctions=[splitWords,puncCount,remove_punctuation,avgWordLength,isFirstPerson,posTag,negation,propNounConcat,\
                 lower_words,removeStopWords,stemming]

    

    for i,current_path_key in enumerate(my_path_keys):
        
        current_path=my_paths[i]
        prepare_data(current_path,inputFilenames,outputFilenames,myFunctions)


def create_Train_Test_data(my_paths,my_path_keys,myindex=(None,None),addWords_bool=True):
    inputFiles=["puncCount.json","avgWordLength.json","isFirstPerson.json","posTag.json"]
    results={}
    for i,current_path_key in enumerate(my_path_keys):

        current_path=my_paths[i]

        inputFName1=current_path+"stemming.json"
        inputFName2=current_path+"kaggle.json" 
        df_features,ratings_words=addTFIDF(inputFName1,inputFName2,myindex=myindex[i])
        addFeatures(df_features,inputFiles,current_path,myindex=myindex[i])
        if addWords_bool:
            if current_path_key=='train':
                all_words=addWords(df_features,ratings_words,1000,myindex[i])['all_words']
                df_words=addWords(df_features,ratings_words,1000,myindex[i])['words_df']
            if current_path_key=='test':
                df_words=addWords(df_features,ratings_words,1000,myindex[i],all_words)['words_df']


        df=pd.DataFrame()
        cols=[elt for elt in df_features.columns if not elt=='words']
        df_features=df_features[cols]
        df[df_features.columns]=df_features

        
        if addWords_bool:
            
            cols=[elt for elt in df_words.columns if not elt=='rating_']
            df[cols]=df_words[cols]
            df_dictionary= transform_csv(df_words, target_col='rating_')
            (X, y)=transform_sklearn_dictionary(df_dictionary)
            clf = MultinomialNB()
            clf.fit(X, y)
            for i in range(5):
                df['bayes_'+str(i)]=clf.predict_proba(X)[:,i]

        results[current_path_key]=df

    return results


