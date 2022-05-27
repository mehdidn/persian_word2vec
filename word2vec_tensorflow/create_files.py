from word_vectors import WordVectors
import numpy as np
import io

out_path = '../out_word2vec_tensorflow'
out_words_vectors = out_path + '/syn0_final.npy'
vocab_path = out_path + '/vocab.txt'

syn0_final = np.load(out_words_vectors)
vocab_words = []
with open(vocab_path) as f : vocab_words = [l.strip() for l in f]
wv = WordVectors(syn0_final, vocab_words)

out_wv = io.open(out_path + '/words-vectors.txt', 'w', encoding='utf-8')
out_v = io.open(out_path + '/vecs.tsv', 'w', encoding='utf-8')
out_m = io.open(out_path + '/meta.tsv', 'w', encoding='utf-8')

for index in range(len(vocab_words)):
	out_wv.write(vocab_words[index] + " " + ' '.join([str(x) for x in syn0_final[index]]) + "\n")
	out_v.write('\t'.join([str(x) for x in syn0_final[index]]) + "\n")
	out_m.write(vocab_words[index] + "\n")

out_wv.close()
out_v.close()
out_m.close()
