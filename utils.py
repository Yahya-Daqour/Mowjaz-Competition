import re

import tensorflow as tf
from tensorflow import keras

import pyarabic.araby as araby
from keras.optimizers import Adam
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import MaxPooling1D, Embedding
from keras.models import Model


def create_model(max_sentence_len, embedd_size, num_labels=10):
    sentence = keras.Input(shape=(max_sentence_len, embedd_size), name='sentence')
    label = keras.Input(shape=(num_labels,), name='label')
    forward_layer = keras.layers.GRU(embedd_size)
    backward_layer = keras.layers.GRU(embedd_size, go_backwards=True)
    rnn = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)
    logits = rnn(sentence)
    
    logits = keras.layers.Dropout(0.2)(logits)
    logits = Dense(embedd_size, activation=tf.nn.sigmoid)(logits)
    logits = keras.layers.Dropout(0.2)(logits)
    logits = keras.layers.Dense(embedd_size, activation=tf.nn.sigmoid)(logits)
    #logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(num_labels, activation=tf.nn.sigmoid)(logits)

    model = keras.Model(sentence, outputs=logits)
    optimizer = Adam(lr=5e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


#def create_model1(max_sentence_len, embedd_size, num_labels=10):
 #   sentence = keras.Input(shape=(max_sentence_len, embedd_size), name='sentence')
  #  label = keras.Input(shape=(num_labels,), name='label')

    # optimizer = Adam(lr=9e-4)
    # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'] )


def normalize_text(text):
    text = text.strip()
    text = araby.strip_tashkeel(text)
    text = ' '.join(araby.tokenize(text))


    # remove extra spaces
    text = re.sub(' +', ' ', text)
    # remove html tags
    text = re.sub(re.compile('<.*?>'), ' ', text)
    # remove twitter usernames, web addresses
    text = re.sub(r"#[\w\d]*|@[.]?[\w\d]*[\'\w*]*|https?:\/\/\S+\b|"
                  r"www\.(\w+\.)+\S*|", '', text)
    # strip repeated chars (extra vals)
    text = re.sub(r'(.)\1+', r"\1\1", text)
    # separate punctuation from words and remove not included marks
    text = " ".join(re.findall(r"[\w']+|[?!,;:]", text))
    # remove underscores
    text = text.replace('_', ' ')
    # remove double quotes
    text = text.strip('\n').replace('\"', '')
    # remove single quotes
    text = text.replace("'", '')
    # remove numbers
    text = ''.join(i for i in text if not i.isdigit())
    # remove extra spaces
    text = re.sub(' +', ' ', text)
    return text