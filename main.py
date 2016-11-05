#!/usr/bin/env python3.5

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import text_to_word_sequence
import testDataset

def createModel(ntags=5,numOfIntermediateLayers=1):
    model = Sequential()
    model.add(Dense(ntags, input_dim=3)) # The word before, the tag before and the current word
    model.add(Activation('tanh'))
    for i in range(0,numOfIntermediateLayers):
        model.add(Dense(output_dim=ntags))
        model.add(Activation('tanh'))
        
    model.add(Dense(output_dim=ntags))
    model.add(Activation("softmax"))
        
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy','categorical_accuracy'])
    return model

testDataset.parseData()
print("Tags: ")
print(testDataset.tags);
model = createModel(len(testDataset.tags),5)
print("Training model...")
model.fit(testDataset.trainX, testDataset.trainY, nb_epoch=5, batch_size=32)
print("Evaluating: ")
loss_and_metrics = model.evaluate(testDataset.testX, testDataset.testY, batch_size=32)
print (loss_and_metrics)


