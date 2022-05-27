import re
import hazm

RE_REMAIN = '[^آ-ی\n\،\؛\؟\!\:\«\»\.\ \‌]'
RE_PERSIAN_ALPHABET = '[^آ-ی]'
RE_JUST_WORDS = '[^آ-ی\n\ \‌]'

read_stop_words = [i.strip().split() for i in open("stopwords.dat").readlines()]
stop_words = []
for w in read_stop_words:
    stop_words.append(w[0])


def norm_words(words):
    words_filtered = []
    for w in words[0]:
        words_filtered.append(w[0])

    return words_filtered


def remove_stopwords(words):
    words_filtered = []
    for w in words:
        if w not in stop_words:
            words_filtered.append(w)

    return words_filtered


def stem(words):
    stemmer = hazm.Stemmer()
    words_stemmed = []
    for w in words:
        words_stemmed.append(stemmer.stem(w))

    return words_stemmed


def lemmatize(words):
    lemmatizer = hazm.Lemmatizer()
    words_lemmatized = []
    for w in words:
        words_lemmatized.append(lemmatizer.lemmatize(w))

    return words_lemmatized


def write_preprocessed_file(file_path, words):
    file_path = file_path.replace('.txt', '') + '_preprocessed.txt'
    file = open(file_path, 'a')
    print(*words, sep=' ', file=file)
    file.close()


def preprocess(file_path):
    file = open(file_path, 'r')
    lines = file.readlines()

    for line in lines:
        line = re.sub(RE_REMAIN, ' ', line)

        # just words
        #line = re.sub(RE_JUST_WORDS, ' ', line)

        check_line = re.sub(RE_PERSIAN_ALPHABET, '', line)
        if len(check_line) == 0:
            continue

        informal_normalizer = hazm.InformalNormalizer(seperation_flag=True)
        words = informal_normalizer.normalize(line)

        words_filtered = norm_words(words)

        # remove stop words
        # in persian
        #words_filtered = remove_stopwords(words_filtered)

        # stemming words
        # in persian
        #words_filtered = stem(words_filtered)

        # lemmatizing words
        # in persian
        #words_filtered = lemmatize(words_filtered)

        write_preprocessed_file(file_path, words_filtered)


data_path = '../Data'
file_paths = [data_path + '/fa.fooladvand.txt', data_path + '/voa_fa_2003-2008_orig.txt']
preprocess(file_paths[0])
