# persian_word2vec
 
visualization for some tests on https://projector.tensorflow.org/

word2vec_tensorflow:

arch = 'skip_gram'
algm = 'negative_sampling'
epochs = 1000
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

visualization:
https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/mehdidn/dfb8ed30d60b0862c8a9c1cc6b4a8c6e/raw/443c8faf04d924f884bfcf1e29af14800a240cc9/word2vec_tensorflow_projector

word2vec_pytorch:

emb_dimension=100
batch_size=1024
window_size=2
negatives=2
iteration=1000
initial_lr=0.025
min_count=3

visualization:
https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/mehdidn/fbbf5cb443eb50751c6a1bea9306bead/raw/805a17f300285aa454f408a853359ccbeff67054/word2vec_pytorch_projector

word2vec_pytorch:

emb_dimension=100
batch_size=256
window_size=5
negatives=5
iteration=100
initial_lr=0.025
min_count=5

visualization:
https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/mehdidn/8bdf6eb3d228bd521cd8b7e1bd2f8729/raw/bd8d491a24abdc0a55a016f2adaeb29473d2b8ea/word2vec_pytorch_projector_min_count_5

for running web app see 'how to run web app.txt':
https://github.com/mehdidn/persian_word2vec/blob/main/how%20to%20run%20web%20app.txt
