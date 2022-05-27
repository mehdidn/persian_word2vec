from word_vectors import WordVectors
import numpy as np
import random

out_path = '../out_word2vec_tensorflow'
out_words_vectors = out_path + '/syn0_final.npy'
vocab_path = out_path + '/vocab.txt'

num_words = 10
num_similar_words = 20
syn0_final = np.load(out_words_vectors)
vocab_words = []
with open(vocab_path) as f : vocab_words = [l.strip() for l in f]
wv = WordVectors(syn0_final, vocab_words)

for i in range(num_words):
    word = random.choice(vocab_words)
    print('nearest neighbors to ' + word + ':')
    most_similar = wv.most_similar(word, num_similar_words)
    for j in range(num_similar_words):
        print(most_similar[j])
    print('-------------------------------\n')