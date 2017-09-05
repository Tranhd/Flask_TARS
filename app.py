from flask import Flask, render_template, jsonify, json
import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

i2w = json.load(open("i2w.txt"))
w2i = json.load(open("w2i.txt"))
vocab_size = json.load(open("vocab_size.txt"))
embedding_size=212
hidden_nodes=412
keep_prob=0.6
learning_rate = 1e-3
unknown_token = "UNK"
sentence_start_token = "GO"
sentence_end_token = "END"
PAD = "PAD"

def format(string_array):
	string = ""
	try:
		string += string_array[0]
	except:
		return "Try again"
	string = string[0].upper() + string[1:]
	next_upper = False
	for i in range(1,len(string_array)):
		word = string_array[i]
		if word == "``": 
			word = '"'
		if word == 'i':
			word = "I"
		if word in [',', '.', '!', '?', "n't", '...',':', '#','$'] or word[0] == "'":
			string += word
			if word in ['.','?','!',':']:
				next_upper = True
		else:
			if next_upper:
				string += " " + word[0].upper() + word[1:] 
				next_upper = False 	
			else:
				string += " " + word
	return string

def prediction():
	tf.reset_default_graph()
	with tf.Session() as sess:
		# Input placeholders 
		with tf.variable_scope('input'):
			seqlen = tf.placeholder(tf.int32, [None])
			x = tf.placeholder(tf.float32, [None, None, vocab_size])

		# Embedding layer 
		with tf.variable_scope('embedding'):
			embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
			embedding_output = tf.nn.embedding_lookup(embeddings, tf.argmax(x, axis=2)) 

		# Cell definition
		with tf.variable_scope('LSTM_cell'):
			cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_nodes, state_is_tuple=True)
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

		# RNN
		with tf.variable_scope('LSTM'):
			outputs, states  = tf.nn.bidirectional_dynamic_rnn(
			                                    cell_fw=cell,
			                                    cell_bw=cell,
			                                    dtype=tf.float32,
			                                    sequence_length=seqlen,
			                                    inputs=embedding_output)
			output_fw, output_bw = outputs
			states_fw, states_bw = states
			output = tf.reshape(output_fw, [-1, hidden_nodes])

		with tf.variable_scope('logits'):
			w = tf.Variable(tf.truncated_normal([hidden_nodes, vocab_size], -0.5, 0.5))
			b = tf.Variable(tf.zeros([vocab_size]))
			logits = tf.matmul(output, w) + b

		with tf.variable_scope('predicion'):
			prediction = tf.nn.softmax(logits)

		with tf.variable_scope('cost'):
			y = tf.placeholder(tf.int32, [None, None, vocab_size])
			labels = tf.reshape(y, [-1, vocab_size])
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

		with tf.variable_scope('train'):
			optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

		with tf.variable_scope('logging'):
		    tf.summary.scalar('current_cost', loss)
		    tf.summary.scalar('current preplexity', tf.exp(loss))
		    summary = tf.summary.merge_all()

		saver = tf.train.Saver()
		saver.restore(sess, tf.train.latest_checkpoint('./model/'))
		max_len = 20
		new_joke = np.zeros((1, max_len, vocab_size))
		new_joke[0,0,w2i[sentence_start_token]]=1
		sentance_length = 1
		while not np.argmax(new_joke[0,sentance_length-1,:]) == w2i[sentence_end_token] and sentance_length<max_len:
			new_wordprobs = prediction.eval({x: new_joke[:,0:sentance_length,:],
				seqlen: [sentance_length]})
			try:
				samples = np.random.multinomial(1, new_wordprobs[-1])
				sampled_word = np.argmax(samples)
			except:
				sampled_word = np.argmax(new_wordprobs[-1])
			if i2w[sampled_word] != unknown_token:
				new_joke[0,sentance_length,sampled_word] = 1
				sentance_length +=1
		string = np.array([])
		for j in new_joke[0]:
			if np.count_nonzero(j) >= 1:
				string = np.append(string, i2w[np.argmax(j)])
	return (format(string[1:-1]))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template("index.html", joke = "Generate joke by clicking button")

@app.route('/about')
def about():
	return render_template("about.html")

@app.route('/generatejoke', methods = ["POST"])
def generateJoke():
	joke_predict = prediction()
	return render_template("index.html", joke=joke_predict)

if __name__ == '__main__':
   app.run()