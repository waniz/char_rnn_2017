import numpy as np
import re
import random
import sys
from keras.models import Sequential
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Activation
from keras.layers import LSTM


class CharRNN:

    # global params
    MAXLEN = 40
    STEP = 1
    BATCH_SIZE = 100

    # model params
    neuron_layers = [128, 128, 128]
    dropout_layers = [0.5, 0.5]

    def __init__(self, file_):
        raw_text = open(file_, encoding="utf-8").read()
        raw_text = raw_text.lower()
        self.raw_text_ru = re.sub("[^а-я,\n .:!?-]", "", raw_text)
        self.chars = sorted(list(set(self.raw_text_ru)))
        self.n_chars = len(raw_text)
        self.n_vocab = len(self.chars)
        self.sentences = []
        self.next_chars = []
        self.model = Sequential()
        self.epoch = 0
        self.X, self.y = None, None

    def get_sentences(self):
        self.sentences = []
        self.next_chars = []
        for i in range(0, len(self.raw_text_ru) - self.MAXLEN, self.STEP):
            self.sentences.append(self.raw_text_ru[i: i + self.MAXLEN])
            self.next_chars.append(self.raw_text_ru[i + self.MAXLEN])
        print(len(self.sentences))
        # self.sentences = self.sentences[:680000]
        # self.sentences = self.sentences[:1360000]
        self.sentences = self.sentences[:400000]
        print(len(self.sentences))

    @staticmethod
    def sample(a, temperature=1.0):
        a = np.log(a) / temperature
        a = np.exp(a) / np.sum(np.exp(a))
        if sum(a) > 1.0:
            a *= 1 - (sum(a) - 1)
            if sum(a) > 1.0:
                a *= 0.99999
        return np.argmax(np.random.multinomial(1, a, 1))

    def vectorization(self):
        char_to_int = dict((c, i) for i, c in enumerate(self.chars))

        self.X = np.zeros((len(self.sentences), self.MAXLEN, len(self.chars)), dtype=np.bool)
        self.y = np.zeros((len(self.sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                self.X[i, t, char_to_int[char]] = 1
            self.y[i, char_to_int[self.next_chars[i]]] = 1

    def build_model(self, previous_save=None):
        self.model.add(LSTM(self.neuron_layers[0], batch_input_shape=(self.BATCH_SIZE, self.MAXLEN, len(self.chars)),
                            return_sequences=True))
        self.model.add(Dropout(self.dropout_layers[0]))
        # self.model.add(LSTM(self.neuron_layers[1], batch_input_shape=(self.BATCH_SIZE, self.MAXLEN, len(self.chars)),
        #                     return_sequences=True))
        # self.model.add(Dropout(self.dropout_layers[1]))
        self.model.add(LSTM(self.neuron_layers[2], batch_input_shape=(self.BATCH_SIZE, self.MAXLEN, len(self.chars)),
                            return_sequences=False))
        self.model.add(Dense(output_dim=len(self.chars)))
        self.model.add(Activation('softmax'))

        if previous_save:
            self.model.load_weights(previous_save)

        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        return self.model

    def train_model(self, from_epoch=0):
        if from_epoch:
            self.epoch = from_epoch

        for iteration in range(0, 10000):
            filepath = "models/weights_ep_%s_loss_{loss:.3f}_val_loss_{val_loss:.3f}.hdf5" % (iteration + self.epoch)
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
            # tensor_board = TensorBoard(log_dir='/home/kustikov/_soft/tutorials/char_rnn/tensorboard_log',
            #                            histogram_freq=1)

            print("==============================================================")
            print("Epoch: ", self.epoch)
            self.model.fit(self.X, self.y, batch_size=self.BATCH_SIZE, nb_epoch=1,
                           callbacks=[checkpoint, reduce_lr],
                           shuffle=False,
                           validation_split=0.1,
                           verbose=1)

    def get_sample(self, temperatures):  # [0.2, 0.5, 1.0]
        start_index = random.randint(0, len(self.raw_text_ru) - self.MAXLEN - 1)
        for T in temperatures:
            print("------------Temperature", T)
            generated = ''
            sentence = self.raw_text_ru[start_index:start_index + self.MAXLEN]
            generated += sentence
            print("Generating with seed: " + sentence)
            print('')

            for i in range(400):
                char_to_int = dict((c, i) for i, c in enumerate(self.chars))
                int_to_char = dict((c, i) for i, c in enumerate(self.chars))

                seed = np.zeros((self.BATCH_SIZE, self.MAXLEN, len(self.chars)))
                for t, char in enumerate(sentence):
                    seed[0, t, char_to_int[char]] = 1

                predictions = self.model.predict(seed, batch_size=self.BATCH_SIZE, verbose=2)[0]
                next_index = self.sample(predictions, T)
                next_char = int_to_char[next_index]

                sys.stdout.write(next_char)
                sys.stdout.flush()

                generated += next_char
                sentence = sentence[1:] + next_char
            print()


rnn_trainer = CharRNN('data/war_and_peace.txt')
rnn_trainer.get_sentences()
rnn_trainer.vectorization()
rnn_trainer.build_model(previous_save=None)
print(rnn_trainer.model.summary())
rnn_trainer.train_model(from_epoch=0)


