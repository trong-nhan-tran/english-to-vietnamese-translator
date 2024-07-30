import json
import numpy as np
import re
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

app = Flask(__name__)

class LuongAttention(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(LuongAttention, self).__init__()
        self.w = layers.Dense(hidden_dim, name='encoder_outputs_dense')

    def call(self, inputs):
        encoder_output_seq, decoder_output = inputs
        z = self.w(encoder_output_seq)
        attention_scores = tf.matmul(decoder_output, z, transpose_b=True)
        attention_weights = tf.keras.activations.softmax(attention_scores, axis=-1)
        context = tf.matmul(attention_weights, encoder_output_seq)
        return attention_weights, context

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, name='encoder_embedding_layer')
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True, name='encoder_lstm')

    def call(self, input):
        embeddings = self.embedding(input)
        output_seq, state_h, state_c = self.lstm(embeddings)
        return output_seq, state_h, state_c

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding_layer = layers.Embedding(vocab_size, embedding_dim, name='decoder_embedding_layer')
        self.lstm = layers.LSTM(hidden_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        self.attention = LuongAttention(hidden_dim)
        self.w = tf.keras.layers.Dense(hidden_dim, activation='tanh', name='attended_outputs_dense')
        self.dense = layers.Dense(vocab_size, name='decoder_dense')

    def call(self, inputs):
        decoder_input, encoder_output_seq, lstm_state = inputs
        embeddings = self.embedding_layer(decoder_input)
        decoder_output, state_h, state_c = self.lstm(embeddings, initial_state=lstm_state)
        weights, context = self.attention([encoder_output_seq, decoder_output])
        decoder_output_with_attention = self.w(tf.concat([tf.squeeze(context, 1), tf.squeeze(decoder_output, 1)], -1))
        logits = self.dense(decoder_output_with_attention)
        return logits, state_h, state_c, weights

with open('./models/tokenizer/source_tokenizer.json') as f:
    data = json.load(f)
    source_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

with open('./models/tokenizer/target_tokenizer.json') as f:
    data = json.load(f)
    target_tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

source_vocab_size = len(source_tokenizer.word_index) + 1
target_vocab_size = len(target_tokenizer.word_index) + 1

embedding_dim = 128
hidden_dim = 256

encoder = Encoder(source_vocab_size, embedding_dim, hidden_dim)
decoder = Decoder(target_vocab_size, embedding_dim, hidden_dim)

encoder.load_weights("./models/luong_attention_weights_new/attention_encoder_weights_with_dropout_ckpt")
decoder.load_weights("./models/luong_attention_weights_new/attention_decoder_weights_with_dropout_ckpt")

max_encoding_len = 48

def preprocess_sentence(s):
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = s.strip()
    s = s.lower()
    return s

def translate_with_attention(sentence, source_tokenizer, encoder, target_tokenizer, decoder, max_translated_len=50):
    input_seq = source_tokenizer.texts_to_sequences([sentence])
    tokenized = source_tokenizer.sequences_to_texts(input_seq)
    input_seq = pad_sequences(input_seq, maxlen=max_encoding_len, padding='post')
    encoder_output, state_h, state_c = encoder.predict(input_seq)
    current_word = '<sos>'
    decoded_sentence = []

    while len(decoded_sentence) < max_translated_len:
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = target_tokenizer.word_index[current_word]
        logits, state_h, state_c, _ = decoder.predict([target_seq, encoder_output, (state_h, state_c)])
        current_token_index = np.argmax(logits[0])
        current_word = target_tokenizer.index_word[current_token_index]
        if current_word == '<eos>':
            break
        decoded_sentence.append(current_word)

    return tokenized[0], ' '.join(decoded_sentence)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    sentence = request.form['sentence']
    preprocessed_sentence = preprocess_sentence(sentence)
    _, translated_sentence = translate_with_attention(preprocessed_sentence, source_tokenizer, encoder, target_tokenizer, decoder)
    return render_template('index.html', original_sentence=sentence, translated_sentence=translated_sentence)

if __name__ == '__main__':
    app.run(debug=True)
