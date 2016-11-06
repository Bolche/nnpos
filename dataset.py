#!/usr/bin/env python3.5
import numpy as np
import re
import random,math

# Start and end symbols
wordDic = {'^':0, '$':1}
words = ['^','$']
tagDic = {'^':0, '$':1}
tags = ['^','$']

trainX = []
trainY = []
testX = []
testY = []

def parseData(testProportion = 0.15):
    global trainX, trainY
    global testX, testY
    global words, worldDic
    global tags, tagDic
    print("Parsing data")
    sentences = []
    with open("data.txt","r") as f:
        for line in f:
            line = line.strip()
            if line.strip() == '':
                continue
            words = re.split(" ", line)
            wordTag = [re.split("/", entry, 2) for entry in words]
            curSent = []
            for pair in wordTag:
                if len(pair) != 2:
                    continue
                word = pair[0].lower()
                tag = pair[1].upper()
                curSent.append( (word, tag) )
                # Fill the word and tag Dic
                wordId = wordDic.get(word)
                tagId = tagDic.get(tag)
                if (wordId == None):
                    words.append(word)
                    wordId = len(words) - 1
                    wordDic[word] = wordId
                if (tagId == None):
                    tags.append(tag)
                    tagId = len(tags) - 1
                    tagDic[tag] = tagId
                
                if pair[1] == '.':
                    sentences.append(curSent)
                    curSent = []
            if len(curSent) > 0:
                sentences.append(curSent)
    random.shuffle(sentences)
    splitI = math.ceil(len(sentences)*testProportion)
    testX,testY = sentenceToVector(sentences[:splitI], len(tags))
    trainX,trainY = sentenceToVector(sentences[splitI:], len(tags))
    
def sentenceToVector(sentences, numOfTags):
    X=[]
    Y=[]
    for sentence in sentences:
        prevWordId = 0
        prevTagId = 0
        for wordTag in sentence:
            wordId = wordDic.get(wordTag[0])
            tagId = tagDic.get(wordTag[1])
            X.append([prevWordId, prevTagId, wordId])
            prevWordId = wordId
            yvec = np.zeros(numOfTags)
            yvec[tagId]=1
            Y.append(yvec)
            prevTagId = tagId
    return (np.array(X), np.array(Y))

def getWordId(word):
    global words,wordDic
    wordId = wordDic.get(word)
    if (wordId == None):
        words.append(word)
        wordId = len(words) - 1
        wordDic[word] = wordId
    return wordId

def getTagId(tag):
    global tags, tagDic
    tagId = tagDic.get(tag)
    if (tagId == None):
        tags.append(tag)
        tagId = len(tags) - 1
        tagDic[tag] = tagId

    return tagId
