import re
from mine import data

pat = re.compile(r"\w+(?:-\w+)*(?:'\w+)?")


def main():
    voc_list, vocabulary, words_idx = data()
    print([word for word in voc_list if pat.fullmatch(word) is None])


if __name__ == "__main__":
    main()
