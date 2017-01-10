from nltk.tokenize import sent_tokenize, StanfordTokenizer
import itertools
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping


def data():
    with open("wonderland.txt", "r", encoding="utf-8-sig") as file:
        words = StanfordTokenizer("/home/lan/stanford-postagger-2016-10-31/stanford-postagger.jar") \
            .tokenize(file.read().lower())
    voc_list = sorted(set(words))
    vocabulary = dict(zip(voc_list, itertools.count()))
    words_idx = [vocabulary[word] for word in words]
    return voc_list, vocabulary, words_idx


def build(voc_length, seq_length):
    return Sequential([
        Embedding(voc_length, 128, input_length=seq_length),
        # LSTM(256, return_sequences=True),
        # Dropout(0.2),
        LSTM(256),
        Dropout(0.2),
        Dense(voc_length, activation='softmax')
    ])


seq_length = 100


def main():
    _, vocabulary, words_idx = data()

    X_y_idx = []

    for i in range(len(words_idx) - seq_length):
        X_y_idx.append(words_idx[i:i + seq_length + 1])

    X_y = np.array(X_y_idx, dtype="int32")
    np.random.shuffle(X_y)
    X = X_y[:, :seq_length]
    y = np_utils.to_categorical(X_y[:, seq_length], len(vocabulary))

    model = build(len(vocabulary), seq_length)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    filepath = "mine-weights-improvement-{epoch:02d}-{val_loss:.4f}-smaller.hdf5"
    callbacks_list = [
        ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min'),
        EarlyStopping(patience=2)
    ]

    model.fit(X, y, nb_epoch=100, batch_size=64, callbacks=callbacks_list, validation_split=0.1)


if __name__ == "__main__":
    main()
