import keras, os
from keras.models import Model
from keras.layers import Input, Embedding, Dense
from keras.layers.recurrent import LSTM
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from keras.callbacks import ModelCheckpoint
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np


HIDDEN_UNITS = 10
DEFAULT_BATCH_SIZE = 64
VERBOSE = 1
DEFAULT_EPOCHS = 10


class Encoder:
    """编码器"""
    def __init__(self, config):
        self.config = config
        self.input_embedding = config['input_embedding']
        self.max_input_seq_length = config['max_input_seq_length']
        self.inputs = tf.placeholder(shape=(self.max_input_seq_length, ), dtype=tf.int32)
        embedding = tf.Variable(tf.random_uniform(self.input_embedding.shape, -1, 1))
        encoder_embedding = tf.nn.embedding_lookup(embedding, self.input_embedding, name='encoder_embedding')
        encoder_cell = LSTMCell(num_units=HIDDEN_UNITS, name='encoder_lstm')
        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_embedding, dtype=tf.float32)



class Decoder:
    """解码器"""
    def __init__(self):
        pass


class Seq2SeqSummary:
    """构建seq2seq"""
    model_name = 'seq2seq'
    def __init__(self, config):
        self.config = config
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        # self.input_embedding = config['input_embedding']
        # self.target_embedding = config['target_embedding']
        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.num_input_tokens, output_dim=HIDDEN_UNITS,
                                      input_length=self.max_input_seq_length, name='encoder_enmbedding')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dence')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model = model
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    @staticmethod
    def get_config_file_path(model_dir_path):
        return os.path.join(model_dir_path, '-config.npy')

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return os.path.join(model_dir_path, '-weight.h5')

    def transform_target_encoding(self, corpus):
        corpus = [[self.target_word2idx[w] for w in s]for s in corpus]
        res = []
        for c in corpus:
            if len(corpus) > self.max_target_seq_length:
                c = c[: self.max_target_seq_length - 1] + ['<EOS>']
                res.append(c)
            else:
                res.append(c)
        return np.array(res)

    def transform_input_encoding(self, corpus):
        corpus = [[self.input_word2idx[w] for w in s] for s in corpus ]
        res = [s[: self.max_input_seq_length] for s in corpus]
        return np.array(res)

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch


    def fit(self, x_train, y_train, x_test, y_test, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, model_dir_path=None):
        if not model_dir_path: model_dir_path = './models'
        config_file_path = Seq2SeqSummary.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqSummary.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        f_config = open(config_file_path, 'wb')
        pickle.dump(self.config, f_config)

        X_train = self.transform_input_encoding(x_train)
        X_test = self.transform_input_encoding(x_test)

        Y_train = self.transform_target_encoding(y_train)
        Y_test = self.transform_target_encoding(y_test)
        print('x_train size: {}, y_train size: {}'.format(len(X_train), len(Y_train)))
        print('X_test size: {}, Y_test size: {}'.format(len(X_test), len(Y_test)))

        train_gen = self.generate_batch(X_train, Y_train, batch_size)
        test_gen = self.generate_batch(X_test, Y_test, batch_size)

        train_num_batches = len(X_train) // batch_size
        test_num_batches = len(X_test) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches)
        self.model.save_weights(weight_file_path)
        return history



