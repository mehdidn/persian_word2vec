import io
import numpy as np

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

out_vocab = io.open(out_path + '/vocab.txt', 'w', encoding='utf-8')
out_v = io.open(out_path + '/vecs.tsv', 'w', encoding='utf-8')
out_m = io.open(out_path + '/meta.tsv', 'w', encoding='utf-8')
out_for_graph = io.open(out_path + '/words-vectors_pytorch.txt', 'w', encoding='utf-8')

for index in range(len(all_words)):
    out_vocab.write(all_words[index] + "\n")
    out_v.write('\t'.join([str(x) for x in all_embeddings[index]]) + "\n")
    out_m.write(all_words[index] + "\n")
    out_for_graph.write(all_words[index] + " " + ' '.join([str(x) for x in all_embeddings[index]]) + "\n")

out_vocab.close()
out_v.close()
out_m.close()
out_for_graph.close()
