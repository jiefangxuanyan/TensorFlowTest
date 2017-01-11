import numpy as np
from nltk.tokenize import sent_tokenize, StanfordTokenizer
import itertools


def data():
    with open("wonderland.txt", "r", encoding="utf-8-sig") as file:
        return [word.lower() for word in StanfordTokenizer(
            path_to_jar=r"C:\stanford-postagger-2016-10-31\stanford-postagger.jar",
            options={"normalizeParentheses": "false", "normalizeOtherBrackets": "false"}
        ).tokenize(file.read())]


def glove(glove_file_name: str):
    result = {}
    with open(glove_file_name, "r", encoding="utf-8") as file:
        for line in file:
            word, *vec_str = line.split()
            result[word] = np.array([float(x) for x in vec_str])
    return result


def glove_filter(word_list, glove_dict):
    for word in word_list:
        if word in glove_dict:
            yield word
        else:
            splited_words = word.split("-")
            if all(sp in glove_dict for sp in splited_words):
                first = True
                for sp in splited_words:
                    if first:
                        first = False
                    else:
                        yield "-"
                    yield sp
            else:
                yield word


def data_from_glove():
    glove_dict = glove(r"F:\documents\embedding\glove.42B.300d.txt")
    word_list = data()
    word_list = list(glove_filter(word_list, glove_dict))
    voc_list = sorted(set(word_list))
    vocabulary = dict(zip(voc_list, itertools.count()))
    word_idx = [vocabulary[word] for word in word_list]
    embed = np.array([glove_dict.get(word, np.zeros(300)) for word in voc_list])
    return voc_list, vocabulary, word_idx, embed


def main():
    # glove_write_db(r"F:\documents\embedding\glove.42B.300d.txt")
    # word_list = data()
    # glove_dict = glove(r"F:\documents\embedding\glove.42B.300d.txt")
    pass


if __name__ == "__main__":
    main()
