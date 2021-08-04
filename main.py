from __future__ import print_function
from builtins import range
from embed_classer import embed
from utils import create_model,normalize_text

import os
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import f1_score, jaccard_score

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

#some configuration
MAX_SEQUENCE_LENGTH = 256 #256
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 39
EPOCHS = 6

#Loading data
df_train = pd.read_csv('/home/yahyadaqour/JUST-Mowjaz-Competition-main/Data/train.tsv', sep="\t")
df_test = pd.read_csv('/home/yahyadaqour/JUST-Mowjaz-Competition-main/Data/test_unlabaled.tsv', sep="\t")
df_dev = pd.read_csv('/home/yahyadaqour/JUST-Mowjaz-Competition-main/Data/validation.tsv', sep="\t")

#load in pre-trained word vectors
print('loading word vectors...')
embedd_path = '/home/yahyadaqour/JUST-Mowjaz-Competition-main/models/full_uni_sg_300_twitter.mdl'

#Save model
model_path = '/home/yahyadaqour/JUST-Mowjaz-Competition-main/models/bi_lstm.best.hdf5'

embedder = embed(embedd_path)

X_train = np.array([normalize_text(text) for text in df_train.Article.values])
Y_train = df_train[df_train.columns[1:]].values

X_dev = np.array([normalize_text(text) for text in df_dev.Article.values])
Y_dev = df_dev[df_dev.columns[1:]].values

X_train = embedder.embed_batch(X_train, MAX_SEQUENCE_LENGTH)
X_dev = embedder.embed_batch(X_dev, MAX_SEQUENCE_LENGTH)

earlystopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=model_path,
    monitor='val_accuracy',
    mode='max',
    save_weights_only=True,
    save_best_only=True,
    verbose=1)


model = create_model(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
print(model.summary())
history = model.fit(X_train,
        Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_dev, Y_dev),
        callbacks=[checkpoint_callback, earlystopping_callback])

preds = model.predict(X_dev) > 0.5

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('Epoch')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('accuracy')
plt.xlabel('Epoch')
plt.show()

print("F1 macro:   {}".format(round(f1_score(Y_dev, preds, average="macro"), 3)))
print("F1 micro:   {}".format(round(f1_score(Y_dev, preds, average="micro"), 3)))
print("F1 samples: {}".format(round(f1_score(Y_dev, preds, average="samples"), 3)))

jaccard_samples = jaccard_score(Y_dev, preds, average="samples")
jaccard_macro = jaccard_score(Y_dev, preds, average="macro")
jaccard_micro = jaccard_score(Y_dev, preds, average="micro")

print("Jaccard Macro Score:         {}".format(round(jaccard_macro, 3)))
print("Jaccard Micro Score:         {}".format(round(jaccard_micro, 3)))
print("Jaccard samples Score:       {}".format(round(jaccard_samples, 3)))


X_test = np.array([normalize_text(text) for text in df_test.Article.values])
X_test = embedder.embed_batch(X_test, MAX_SEQUENCE_LENGTH)
preds = model.predict(X_test) > 0.5

df = pd.DataFrame(data=preds, index=None, columns=None, dtype=int)
df.to_csv("/home/yahyadaqour/JUST-Mowjaz-Competition-main/Data/outputs/answer.tsv", header=False, index=False, sep="\t")