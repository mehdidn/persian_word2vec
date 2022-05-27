import tensorflow as tf


class Word2VecModel(tf.keras.Model):
	def __init__(self,
	             unigram_counts,
	             arch='skip_gram',
	             algm='negative_sampling',
	             hidden_size=300,
	             batch_size=256,
	             negatives=5,
	             power=0.75,
	             alpha=0.025,
	             min_alpha=0.0001,
	             add_bias=True,
	             random_seed=0):

		super(Word2VecModel, self).__init__()
		self._unigram_counts = unigram_counts
		self._arch = arch
		self._algm = algm
		self._hidden_size = hidden_size
		self._vocab_size = len(unigram_counts)
		self._batch_size = batch_size
		self._negatives = negatives
		self._power = power
		self._alpha = alpha
		self._min_alpha = min_alpha
		self._add_bias = add_bias
		self._random_seed = random_seed

		self._input_size = (self._vocab_size if self._algm == 'negative_sampling'
		                    else self._vocab_size - 1)

		self.add_weight('syn0',
		                shape=[self._vocab_size, self._hidden_size],
		                initializer=tf.keras.initializers.RandomUniform(
			                minval=-0.5 / self._hidden_size,
			                maxval=0.5 / self._hidden_size))

		self.add_weight('syn1',
		                shape=[self._input_size, self._hidden_size],
		                initializer=tf.keras.initializers.RandomUniform(
			                minval=-0.1, maxval=0.1))

		self.add_weight('biases',
		                shape=[self._input_size],
		                initializer=tf.keras.initializers.Zeros())

	def call(self, inputs, labels):
		if self._algm == 'negative_sampling':
			loss = self._negative_sampling_loss(inputs, labels)
		elif self._algm == 'hierarchical_softmax':
			loss = self._hierarchical_softmax_loss(inputs, labels)
		return loss

	def _negative_sampling_loss(self, inputs, labels):
		_, syn1, biases = self.weights

		sampled_values = tf.random.fixed_unigram_candidate_sampler(
			true_classes=tf.expand_dims(labels, 1),
			num_true=1,
			num_sampled=self._batch_size * self._negatives,
			unique=True,
			range_max=len(self._unigram_counts),
			distortion=self._power,
			unigrams=self._unigram_counts)

		sampled = sampled_values.sampled_candidates
		sampled_mat = tf.reshape(sampled, [self._batch_size, self._negatives])
		inputs_syn0 = self._get_inputs_syn0(inputs)
		true_syn1 = tf.gather(syn1, labels)
		sampled_syn1 = tf.gather(syn1, sampled_mat)
		true_logits = tf.reduce_sum(tf.multiply(inputs_syn0, true_syn1), 1)
		sampled_logits = tf.einsum('ijk,ikl->il', tf.expand_dims(inputs_syn0, 1), tf.transpose(sampled_syn1, (0, 2, 1)))

		if self._add_bias:
			true_logits += tf.gather(biases, labels)
			sampled_logits += tf.gather(biases, sampled_mat)

		true_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(true_logits), logits=true_logits)
		sampled_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

		loss = tf.concat([tf.expand_dims(true_cross_entropy, 1), sampled_cross_entropy], 1)
		return loss

	def _hierarchical_softmax_loss(self, inputs, labels):
		_, syn1, biases = self.weights

		inputs_syn0_list = tf.unstack(self._get_inputs_syn0(inputs))
		codes_points_list = tf.unstack(labels)
		max_depth = (labels.shape.as_list()[1] - 1) // 2
		loss = []
		for i in range(self._batch_size):
			inputs_syn0 = inputs_syn0_list[i]
			codes_points = codes_points_list[i]
			true_size = codes_points[-1]

			codes = codes_points[:true_size]
			points = codes_points[max_depth:max_depth + true_size]
			logits = tf.reduce_sum(tf.multiply(inputs_syn0, tf.gather(syn1, points)), 1)
			if self._add_bias:
				logits += tf.gather(biases, points)

			loss.append(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(codes, 'float32'), logits=logits))
		loss = tf.concat(loss, axis=0)
		return loss

	def _get_inputs_syn0(self, inputs):
		syn0, _, _ = self.weights
		if self._arch == 'skip_gram':
			inputs_syn0 = tf.gather(syn0, inputs)
		else:
			inputs_syn0 = []
			contexts_list = tf.unstack(inputs)
			for i in range(self._batch_size):
				contexts = contexts_list[i]
				context_words = contexts[:-1]
				true_size = contexts[-1]
				inputs_syn0.append(tf.reduce_mean(tf.gather(syn0, context_words[:true_size]), axis=0))
			inputs_syn0 = tf.stack(inputs_syn0)

		return inputs_syn0
