#!/usr/bin/env python3.5

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import text_to_word_sequence
import dataset

def createModel(ntags=5,numOfIntermediateLayers=1):
    model = Sequential()
    model.add(Dense(ntags*3, input_dim=3)) # The word before, the tag before and the current word
    model.add(Activation('tanh'))
    for i in range(0,numOfIntermediateLayers):
        model.add(Dense(output_dim=ntags*3, init='uniform'))
        model.add(Activation('tanh'))
        
    model.add(Dense(output_dim=ntags, init='uniform'))
    model.add(Activation("softmax"))
        
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy','categorical_accuracy'])
    return model

dataset.parseData()
print("Tags: ")
print(dataset.tags);
model = createModel(len(dataset.tags),4)
print("Training model...")
model.fit(dataset.trainX, dataset.trainY, nb_epoch=5, batch_size=32)
print("Evaluating: ")
loss_and_metrics = model.evaluate(dataset.testX, dataset.testY, batch_size=32)
print (loss_and_metrics)

sentence = input("Now, enter your sentence: ")
word_sequence = text_to_word_sequence(sentence)
prev_word = "^"
prev_tag = 0
for word in word_sequence:
    xvec = np.array([[dataset.getWordId(prev_word), dataset.getTagId(prev_tag), dataset.getWordId(word)]])
    predictions = model.predict(xvec)
    tag = predictions.argmax()
    print(word+" - "+dataset.tags[tag])
    prev_tag = tag
    prev_word = word
