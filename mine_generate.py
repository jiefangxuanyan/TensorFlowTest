import numpy.random
import mine


def translate(voc_list, idx):
    return [voc_list[i] for i in idx]


def main():
    voc_list, _, words_idx, embed = mine.data()

    model = mine.build(mine.seq_length, embed)
    model.load_weights("mine-weights-improvement-10-4.3625-smaller.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    generate_length = 200

    begin = numpy.random.randint(0, len(words_idx) - mine.seq_length - generate_length)

    X_idx = words_idx[begin:begin + mine.seq_length]

    for i in range(generate_length):
        X = numpy.reshape(X_idx[len(X_idx) - mine.seq_length:], (1, mine.seq_length))
        Y = model.predict(X, batch_size=1)
        cur = numpy.argmax(Y)
        X_idx.append(cur)

    print(translate(voc_list, X_idx[:mine.seq_length]))
    print(translate(voc_list, X_idx[mine.seq_length:]))
    print(translate(voc_list, words_idx[begin + mine.seq_length:begin + mine.seq_length + generate_length]))


if __name__ == "__main__":
    main()
