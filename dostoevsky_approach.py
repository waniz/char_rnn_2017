import numpy as np
import re

from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.optimizers import RMSprop


class NBatchLogger(Callback):

    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs={}):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            # you can access loss, accuracy in self.params['metrics']
            # print('\n{}/{} - loss ....\n'.format(self.seen, self.params['nb_sample']))
            print('\n Batch Loss: {0}'.format(self.params['metrics']))


class CharRNN:

    # global params
    MAXLEN = 20
    STEP = 1
    BATCH_SIZE = 1000

    VALIDATION_SPLIT_GEN = 0.95
    GENERATOR_TRAINING = True

    # model params
    neuron_layers = [400, 400, 300]
    dropout_layers = [0.4, 0.4]
    # dense_layers = [320]

    def __init__(self, file_, generator_training_type=False):
        raw_text = open(file_, encoding="utf-8").read()
        raw_text = raw_text.lower()
        self.raw_text_ru = re.sub("[^а-я, .]", "", raw_text)
        self.chars = sorted(list(set(self.raw_text_ru)))
        self.n_chars = len(raw_text)
        self.n_vocab = len(self.chars)
        self.sentences = []
        self.next_chars = []
        self.model = Sequential()
        self.epoch = 0
        self.X, self.y = None, None

        self.validation_set = self.raw_text_ru[int(len(self.raw_text_ru) * self.VALIDATION_SPLIT_GEN):]
        self.raw_text_ru = self.raw_text_ru[:int(len(self.raw_text_ru) * self.VALIDATION_SPLIT_GEN)]

        with open('data/Dostoevsky_val.txt', 'w') as file:
            file.write(self.validation_set)

        print('Corpus train length: ', len(self.raw_text_ru))
        print('Corpus val length  : ', len(self.validation_set))

        self.GENERATOR_TRAINING = generator_training_type

    def get_sentences(self):
        self.sentences = []
        self.next_chars = []
        for i in range(0, len(self.raw_text_ru) - self.MAXLEN, self.STEP):
            self.sentences.append(self.raw_text_ru[i: i + self.MAXLEN])
            self.next_chars.append(self.raw_text_ru[i + self.MAXLEN])
        print('Corpus length: ', len(self.sentences))

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
        self.model.add(LSTM(self.neuron_layers[0],
                            batch_input_shape=(self.BATCH_SIZE, self.MAXLEN, len(self.chars)),
                            return_sequences=True))
        self.model.add(Dropout(self.dropout_layers[0]))

        if self.neuron_layers[1]:
            self.model.add(LSTM(self.neuron_layers[1],
                                batch_input_shape=(self.BATCH_SIZE, self.MAXLEN, len(self.chars)),
                                return_sequences=True))
            self.model.add(Dropout(self.dropout_layers[1]))

        self.model.add(LSTM(self.neuron_layers[2],
                            batch_input_shape=(self.BATCH_SIZE, self.MAXLEN, len(self.chars)),
                            return_sequences=False))

        # self.model.add(Dense(self.dense_layers[0]))
        self.model.add(Dense(output_dim=len(self.chars)))
        self.model.add(Activation('softmax'))

        if previous_save:
            self.model.load_weights(previous_save)

        rmsprop = RMSprop(lr=0.001)  # lr=0.001 till 25- epochs
        self.model.compile(loss='categorical_crossentropy', optimizer=rmsprop)

        model_json = self.model.to_json()
        with open('models_dostoevsky/current_model.json', 'w') as json_file:
            json_file.write(model_json)

        return self.model

    def train_model(self, from_epoch=0):
        if from_epoch:
            self.epoch = from_epoch

        for iteration in range(0, 10000):
            filepath = "models_dostoevsky/weights_ep_%s_loss_{loss:.3f}_val_loss_{val_loss:.3f}.hdf5" % (iteration + self.epoch)
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
            # logger_ = NBatchLogger(display=1000)

            print("==============================================================")
            print("Epoch: ", self.epoch)

            self.model.fit(self.X, self.y, batch_size=self.BATCH_SIZE, nb_epoch=1,
                           callbacks=[checkpoint, reduce_lr],
                           shuffle=False,
                           validation_split=0.1,
                           verbose=1)

    """ helpers for train model with fit_generator """

    def generate_text_slices_val(self):
        text = self.validation_set
        yield len(text), text[:self.MAXLEN]

        while True:
            for i in range(0, len(text) - self.MAXLEN, self.STEP):
                sentence = text[i: i + self.MAXLEN]
                next_char = text[i + self.MAXLEN]
                yield sentence, next_char

    def generate_text_slices(self):
        text = self.raw_text_ru
        yield len(text), text[:self.MAXLEN]

        while True:
            for i in range(0, len(text) - self.MAXLEN, self.STEP):
                sentence = text[i: i + self.MAXLEN]
                next_char = text[i + self.MAXLEN]
                yield sentence, next_char

    def generate_arrays_from_data(self, train=True):

        char_to_int = dict((c, i) for i, c in enumerate(self.chars))

        if train:
            slices = self.generate_text_slices()
        else:
            slices = self.generate_text_slices_val()

        text_len, seed = next(slices)
        samples = (text_len - self.MAXLEN + self.STEP - 1) / self.STEP
        yield samples, seed

        while True:
            X = np.zeros((self.BATCH_SIZE, self.MAXLEN, len(self.chars)), dtype=np.bool)
            y = np.zeros((self.BATCH_SIZE, len(self.chars)), dtype=np.bool)
            for i in range(self.BATCH_SIZE):
                sentence, next_char = next(slices)
                for t, char in enumerate(sentence):
                    X[i, t, char_to_int[char]] = 1
                y[i, char_to_int[next_char]] = 1
            yield X, y

    """ helpers for train model with fit_generator """

    def train_model_generator(self, from_epoch=0):
        train_generator = self.generate_arrays_from_data(train=True)
        samples, seed = next(train_generator)
        print('samples per epoch %s' % samples)
        last_epoch = from_epoch

        self.model.metadata = {'epoch': 0, 'loss': [], 'val_loss': []}

        for epoch in range(last_epoch + 1, last_epoch + 10000):
            val_gen = self.generate_arrays_from_data(train=False)
            val_samples, _ = next(val_gen)

            filepath = "models_dostoevsky/weights_ep_%s_loss_{loss:.3f}_val_loss_{val_loss:.3f}.hdf5" % epoch
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=1, min_lr=0.0001)

            self.model.fit_generator(train_generator, validation_data=val_gen,
                                     nb_val_samples=val_samples,
                                     samples_per_epoch=samples,
                                     nb_epoch=1, max_q_size=10,
                                     callbacks=[checkpoint, reduce_lr], verbose=1)


rnn_trainer = CharRNN('data/dost_best.txt', generator_training_type=True)

if rnn_trainer.GENERATOR_TRAINING:
    rnn_trainer.build_model(previous_save=None)
    print(rnn_trainer.model.summary())
    rnn_trainer.train_model_generator(from_epoch=0)
else:
    rnn_trainer.get_sentences()
    rnn_trainer.vectorization()
    rnn_trainer.build_model(previous_save=None)
    print(rnn_trainer.model.summary())
    rnn_trainer.train_model(from_epoch=0)



