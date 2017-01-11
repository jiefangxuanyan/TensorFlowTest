from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle


def build(seq_length: int, embed: np.array):
    model = Sequential([
        Embedding(embed.shape[0], embed.shape[1], input_length=seq_length, weights=[embed]),
        # LSTM(64, return_sequences=True),
        # Dropout(0.2),
        LSTM(128),
        Dropout(0.2),
        Dense(embed.shape[0], activation='softmax')
    ])
    model.layers[0].trainable = False
    return model


seq_length = 100


def main():
    with open("glove_cache.pickle", "rb") as fin:
        _, vocabulary, words_idx, embed = pickle.load(fin)

    X_y_idx = []

    for i in range(len(words_idx) - seq_length):
        X_y_idx.append(words_idx[i:i + seq_length + 1])

    X_y = np.array(X_y_idx, dtype="int32")
    np.random.shuffle(X_y)
    X = X_y[:, :seq_length]
    y = np_utils.to_categorical(X_y[:, seq_length], len(vocabulary))

    model = build(len(vocabulary), embed)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    filepath = "mine-weights-improvement-{epoch:02d}-{val_loss:.4f}-smaller.hdf5"
    callbacks_list = [
        ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='min'),
        EarlyStopping(patience=2)
    ]

    model.fit(X, y, nb_epoch=100, batch_size=64, callbacks=callbacks_list, validation_split=0.1)


if __name__ == "__main__":
    main()
