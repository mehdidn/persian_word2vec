from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random

out_path = '../out_word2vec_pytorch'
out_words_vectors = out_path + '/words-vectors.txt'
f = open(out_words_vectors)
f.readline()
all_embeddings = []
all_words = []
word2id = dict()

for i, line in enumerate(f):
    line = line.strip().split(' ')
    word = line[0]
    embedding = [float(x) for x in line[1:]]
    assert len(embedding) == 100
    all_embeddings.append(embedding)
    all_words.append(word)
    word2id[word] = i

all_embeddings = np.array(all_embeddings)

for i in range(10):
    word = random.choice(all_words)
    print('nearest neighbors to ' + word + ':')
    wid = word2id[word]

    embedding = all_embeddings[wid:wid+1]

    d = cosine_similarity(embedding, all_embeddings)[0]
    d = zip(all_words, d)
    d = sorted(d, key=lambda x: x[1], reverse=True)

    for w in d[1:21]:
        if len(w[0]) < 2:
            continue
        print(w)

    print('-------------------------------\n')
