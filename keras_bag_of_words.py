#!/usr/bin/env python

import json
import random
import numpy as np

from keras.models import Sequential, Graph
from keras.preprocessing import text
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l1, l2
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM

SEQ_LENGTH = 64
BATCH_SIZE = 16
HIDDEN_SIZE = 256
NUM_WORDS = 10000

examples = [json.loads(line) for line in open('topic-sample.json').readlines() if line]
random.seed(1234)
random.shuffle(examples)
abstracts = [ e['abstract'].encode('utf8') for e in examples ]
titles = [ e['title'].encode('utf8') for e in examples ]

def make_data():
    abstractWords = [a.split() for a in abstracts]
    titleWords = [t.split() for t in titles]
    newAbstracts = [' '.join(['a:' + w for w in a]) for a in abstractWords]
    newTitles = [' '.join(['t:' + w for w in t]) for t in titleWords]
    return [a + ' ' + t for (a, t) in zip(newAbstracts, newTitles)]

data = make_data()
tokenizer = Tokenizer(nb_words=NUM_WORDS, split=' ')
tokenizer.fit_on_texts(data)

vocab_size = tokenizer.nb_words

def build_lr_model():
    model = Sequential()
    model.add(Dense(2, input_dim=vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    return model

def build_mlp():
    model = Sequential()
    model.add(Dense(HIDDEN_SIZE, input_dim=vocab_size))
    model.add(Dense(HIDDEN_SIZE))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    return model

def build_lstm():
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(LSTM(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    return model

print 'Building model.'
training_model = build_mlp()
print '... done'

X = tokenizer.texts_to_matrix(data, "binary")
Y = np.zeros((len(data), 2))
for i, ex in enumerate(examples):
    if ex['topic'] == 'compsci': Y[i,0] = 1
    else: Y[i, 1] = 1

training_model.fit(X, Y, batch_size=BATCH_SIZE, show_accuracy=True,
                   verbose=True, nb_epoch=3, validation_split=0.1)


# vim: et sw=4 sts=4
