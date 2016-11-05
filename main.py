#!/usr/bin/env python3.5

from keras.models import Sequential
import testDataset

def createModel(ntags=5,numOfIntermediateLayers=1):
    model = Sequential()
    model.add(Dense(ntags, input_dim=3)) # The current word, the word before and the word after
    model.add(Activation('tanh'))
    for i in range(0,numOfIntermediateLayers):
        model.add(Dense(ntags, input_dim=maxwords))
        model.add(Activation('tanh'))
        
        model.add(Dense(output_dim=ntags))
        model.add(Activation("softmax"))
        
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

