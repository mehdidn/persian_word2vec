import os
import tensorflow as tf
import numpy as np
from absl import app
from absl import flags
from dataset import WordTokenizer
from dataset import Word2VecDatasetBuilder
from model import Word2VecModel
import utils

data_path = '../Data'
in_file_paths = [data_path + '/fa.fooladvand_preprocessed.txt', data_path + '/voa_fa_2003-2008_orig.txt']
out_path = '../out_word2vec_tensorflow'

flags.DEFINE_list('filenames', in_file_paths[0], 'input txt files.')
flags.DEFINE_string('out_dir', out_path, 'Output directory.')
FLAGS = flags.FLAGS


def main(_):
	arch = 'skip_gram'  # skip_gram or cbow
	algm = 'negative_sampling'  # negative_sampling or hierarchical_softmax
	epochs = 1
	batch_size = 1024
	max_vocab_size = 0
	min_count = 3
	sample = 1e-3
	window_size = 2
	hidden_size = 100
	negatives = 2
	power = 0.75
	alpha = 0.025
	min_alpha = 0.0001
	add_bias = True
	log_per_steps = 10000
	file_paths = FLAGS.filenames
	out_dir = FLAGS.out_dir

	tokenizer = WordTokenizer(max_vocab_size=max_vocab_size, min_count=min_count, sample=sample)
	tokenizer.build_vocab(file_paths)

	builder = Word2VecDatasetBuilder(tokenizer,
	                                 arch=arch,
	                                 algm=algm,
	                                 epochs=epochs,
	                                 batch_size=batch_size,
	                                 window_size=window_size)
	dataset = builder.build_dataset(file_paths)
	word2vec = Word2VecModel(tokenizer.unigram_counts,
	                         arch=arch,
	                         algm=algm,
	                         hidden_size=hidden_size,
	                         batch_size=batch_size,
	                         negatives=negatives,
	                         power=power,
	                         alpha=alpha,
	                         min_alpha=min_alpha,
	                         add_bias=add_bias)

	train_step_signature = utils.get_train_step_signature(arch, algm, batch_size, window_size, builder._max_depth)
	optimizer = tf.keras.optimizers.SGD(1.0)

	@tf.function(input_signature=train_step_signature)
	def train_step(inputs, labels, progress):
		loss = word2vec(inputs, labels)
		gradients = tf.gradients(loss, word2vec.trainable_variables)

		learning_rate = tf.maximum(alpha * (1 - progress[0]) + min_alpha * progress[0], min_alpha)

		if hasattr(gradients[0], '_values'):
			gradients[0]._values *= learning_rate
		else:
			gradients[0] *= learning_rate

		if hasattr(gradients[1], '_values'):
			gradients[1]._values *= learning_rate
		else:
			gradients[1] *= learning_rate

		if hasattr(gradients[2], '_values'):
			gradients[2]._values *= learning_rate
		else:
			gradients[2] *= learning_rate

		optimizer.apply_gradients(zip(gradients, word2vec.trainable_variables))

		return loss, learning_rate

	average_loss = 0.
	for step, (inputs, labels, progress) in enumerate(dataset):
		loss, learning_rate = train_step(inputs, labels, progress)
		average_loss += loss.numpy().mean()
		if step % log_per_steps == 0:
			if step > 0:
				average_loss /= log_per_steps
			print('step:', step, 'average_loss:', average_loss,
			      'learning_rate:', learning_rate.numpy())
			average_loss = 0.

	syn0_final = word2vec.weights[0].numpy()
	np.save(os.path.join(out_dir, 'syn0_final'), syn0_final)
	with tf.io.gfile.GFile(os.path.join(out_dir, 'vocab.txt'), 'w') as f:
		for w in tokenizer.table_words:
			f.write(w + '\n')
	print('Word embeddings saved to',
	      os.path.join(out_dir, 'syn0_final.npy'))
	print('Vocabulary saved to', os.path.join(out_dir, 'vocab.txt'))


if __name__ == '__main__':
	app.run(main)
