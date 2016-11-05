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

def getData(testProportion = 0.15):
    global trainX, trainY
    global testX, testY
    print("Parsing data")
    sentences = []
    with open("data/all2.txt","r") as f:
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
                curSent.append( (pair[0], pair[1]) )
                if pair[1] == '.':
                    sentences.append(curSent)
                    curSent = []
            if len(curSent) > 0:
                sentences.append(curSent)
    random.shuffle(sentences)
    splitI = math.ceil(len(sentences)*testProportion)
    testX,testY = sentenceToVector(sentences[:splitI])
    trainX,trainY = sentenceToVector(sentences[splitI:])
    
def sentenceToVector(sentences):
    #TODO
    for sentence in sentences:
        for wordTag in sentence:
            wordId = wordDic.get(wordTag[0])
            tagId = tagDic.get(wordTag[1])
            if (wordId == None):
                words.append(wordTag[0])
                wordId = len(words) - 1
                wordDic[wordTag[0]] = wordId
            if (tagId == None):
                tags.append(wordTag[1])
                tagId = len(tags) - 1
                tagDic[wordTag[0]] = wordId
