import numpy as np
import re

from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, LSTM
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


class CharRNN:

    # global params
    MAXLEN = 20
    STEP = 1
    BATCH_SIZE = 1000

    VALIDATION_SPLIT_GEN = 0.9
    GENERATOR_TRAINING = True

    # model params
    neuron_layers = [128, 128, 128]
    dropout_layers = [0.2, 0.2]
    dense_layers = [128]

    def __init__(self, file_, generator_training_type=False):
        raw_text = open(file_, encoding="utf-8").read()
        raw_text = raw_text.lower()
        self.raw_text_ru = re.sub("[^а-я ]", "", raw_text)
        self.words = list(set(self.raw_text_ru.split()))
        self.list_words = self.raw_text_ru.split()
        self.sentence_helper = []

        print('Length vocabulary: ', len(self.words))

        self.sentences = []
        self.next_words = []
        self.model = Sequential()
        self.epoch = 0
        self.X, self.y = None, None
        self.GENERATOR_TRAINING = generator_training_type

        self.validation_set = self.raw_text_ru[int(len(self.raw_text_ru) * self.VALIDATION_SPLIT_GEN):]
        self.raw_text_ru = self.raw_text_ru[:int(len(self.raw_text_ru) * self.VALIDATION_SPLIT_GEN)]

        with open('data/Lev_Tolstoy_val.txt', 'w') as file:
            file.write(self.validation_set)

    def get_sentences(self):
        for i in range(0, len(self.list_words) - self.MAXLEN, self.STEP):
            self.sentence_helper = ' '.join(self.list_words[i: i + self.MAXLEN])
            self.sentences.append(self.sentence_helper)
            self.next_words.append(self.list_words[i + self.MAXLEN])
        print('Length of sequences: ', len(self.sentences))
        print('Length of next_words: ', len(self.next_words))

    def vectorization(self):
        word_to_int = dict((c, i) for i, c in enumerate(self.words))

        self.sentences = self.sentences[:20000]

        self.X = np.zeros((len(self.sentences), self.MAXLEN, len(self.words)), dtype=np.bool)
        self.y = np.zeros((len(self.sentences), len(self.words)), dtype=np.bool)
        for i, sentence in enumerate(self.sentences):
            for t, word in enumerate(sentence.split()):
                self.X[i, t, word_to_int[word]] = 1
            self.y[i, word_to_int[self.next_words[i]]] = 1

    def build_model(self, previous_save=None):
        self.model.add(LSTM(self.neuron_layers[0],
                            batch_input_shape=(self.BATCH_SIZE, self.MAXLEN, len(self.words)),
                            return_sequences=True))
        self.model.add(Dropout(self.dropout_layers[0]))

        if self.neuron_layers[1]:
            self.model.add(LSTM(self.neuron_layers[1],
                                batch_input_shape=(self.BATCH_SIZE, self.MAXLEN, len(self.words)),
                                return_sequences=True))
            self.model.add(Dropout(self.dropout_layers[1]))

        self.model.add(LSTM(self.neuron_layers[2],
                            batch_input_shape=(self.BATCH_SIZE, self.MAXLEN, len(self.words)),
                            return_sequences=False))

        self.model.add(Dense(self.dense_layers[0]))
        self.model.add(Dense(output_dim=len(self.words)))
        self.model.add(Activation('softmax'))

        if previous_save:
            self.model.load_weights(previous_save)

        # rmsprop = RMSprop(decay=0.05)
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        model_json = self.model.to_json()
        with open('models/current_model.json', 'w') as json_file:
            json_file.write(model_json)

        return self.model

    def train_model(self, from_epoch=0):
        if from_epoch:
            self.epoch = from_epoch

        for iteration in range(0, 10000):
            filepath = "models_word/weights_ep_%s_loss_{loss:.3f}_val_loss_{val_loss:.3f}.hdf5" % (iteration + self.epoch)
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='min')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)

            print("==============================================================")
            print("Epoch: ", self.epoch)

            self.model.fit(self.X, self.y, batch_size=self.BATCH_SIZE, nb_epoch=1,
                           callbacks=[checkpoint, reduce_lr],
                           shuffle=False,
                           validation_split=0.1,
                           verbose=1)

rnn_trainer = CharRNN('data/word_modelling_sample.txt', generator_training_type=True)
rnn_trainer.get_sentences()
rnn_trainer.vectorization()
rnn_trainer.build_model(previous_save=None)
print(rnn_trainer.model.summary())
rnn_trainer.train_model(from_epoch=0)
