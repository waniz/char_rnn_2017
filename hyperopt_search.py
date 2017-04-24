import numpy as np
import pandas as pd
import re
import warnings

from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, LSTM
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

# global params
MAXLEN = 20
STEP = 1
BATCH_SIZE = 1000
CHARS = sorted(list(set(raw_text_ru)))


file_ = 'data/Lev_Tolstoy_all.txt'
raw_text = open(file_, encoding="utf-8").read()
raw_text = raw_text.lower()
raw_text_ru = re.sub("[^а-я, .]", "", raw_text)

# filter ---
val_set = raw_text_ru[50000:60000 + MAXLEN]
raw_text_ru = raw_text_ru[:50000 + MAXLEN]


""" helpers for train model with fit_generator """


def generate_text_slices_val():
    text = val_set
    yield len(text), text[:MAXLEN]

    while True:
        for i in range(0, len(text) - MAXLEN, STEP):
            sentence = text[i: i + MAXLEN]
            next_char = text[i + MAXLEN]
            yield sentence, next_char


def generate_text_slices():
    text = raw_text_ru
    yield len(text), text[:MAXLEN]

    while True:
        for i in range(0, len(text) - MAXLEN, STEP):
            sentence = text[i: i + MAXLEN]
            next_char = text[i + MAXLEN]
            yield sentence, next_char


def generate_arrays_from_data(train=True):
    char_to_int = dict((c, i) for i, c in enumerate(CHARS))

    if train:
        slices = generate_text_slices()
    else:
        slices = generate_text_slices_val()

    text_len, seed = next(slices)
    samples = (text_len - MAXLEN + STEP - 1) / STEP
    yield samples, seed

    while True:
        X = np.zeros((BATCH_SIZE, MAXLEN, len(CHARS)), dtype=np.bool)
        y = np.zeros((BATCH_SIZE, len(CHARS)), dtype=np.bool)
        for i in range(BATCH_SIZE):
            sentence, next_char = next(slices)
            for t, char in enumerate(sentence):
                X[i, t, char_to_int[char]] = 1
            y[i, char_to_int[next_char]] = 1
        yield X, y


def get_sentences():
    sentences = []
    next_chars = []
    for i in range(0, len(raw_text_ru) - MAXLEN, STEP):
        sentences.append(raw_text_ru[i: i + MAXLEN])
        next_chars.append(raw_text_ru[i + MAXLEN])
    print('Corpus train length: ', len(sentences))
    return sentences, next_chars


def get_sentences_val():
    sentences = []
    next_chars = []
    for i in range(0, len(val_set) - MAXLEN, STEP):
        sentences.append(val_set[i: i + MAXLEN])
        next_chars.append(val_set[i + MAXLEN])
    print('Corpus val length: ', len(sentences))
    return sentences, next_chars


def vectorization(sentences, next_chars):
    char_to_int = dict((c, i) for i, c in enumerate(CHARS))

    X = np.zeros((len(sentences), MAXLEN, len(CHARS)), dtype=np.bool)
    y = np.zeros((len(sentences), len(CHARS)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_to_int[char]] = 1
        y[i, char_to_int[next_chars[i]]] = 1

    return X, y
""" helpers for train model with fit_generator """


def hyperopt_search_generator(hpparams):
    all_results = []
    all_results_train = []

    model = Sequential()
    model.add(LSTM(output_dim=round(hpparams['output_dim_1']),
                   batch_input_shape=(BATCH_SIZE, MAXLEN, len(CHARS)), return_sequences=True))
    model.add(Dropout(hpparams['dropout_1']))
    model.add(LSTM(output_dim=round(hpparams['output_dim_2']),
                   batch_input_shape=(BATCH_SIZE, MAXLEN, len(CHARS)), return_sequences=True))
    model.add(Dropout(hpparams['dropout_2']))
    model.add(LSTM(output_dim=round(hpparams['output_dim_3']),
                   batch_input_shape=(BATCH_SIZE, MAXLEN, len(CHARS)), return_sequences=False))
    model.add(Dense(output_dim=round(hpparams['output_dim_4'])))
    model.add(Dense(output_dim=len(CHARS)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    train_generator = generate_arrays_from_data(train=True)
    samples, seed = next(train_generator)

    val_gen = generate_arrays_from_data(train=False)
    val_samples, _ = next(val_gen)

    for epoch in range(hpparams['epochs']):
        print('Training epoch %s of %s' % (epoch, hpparams['epochs']))
        hist = model.fit_generator(train_generator, validation_data=val_gen, nb_val_samples=val_samples,
                                   samples_per_epoch=samples, nb_epoch=1, verbose=1)

        all_results.append(hist.history['val_loss'])
        all_results_train.append(hist.history['loss'])

    return min(all_results), min(all_results_train)


def hyperopt_search(hpparams):
    all_results = []
    all_results_train = []

    print('Params: %s' % hpparams)

    model = Sequential()
    model.add(LSTM(output_dim=round(hpparams['output_dim_1']),
                   batch_input_shape=(BATCH_SIZE, MAXLEN, len(CHARS)), return_sequences=True))
    model.add(Dropout(hpparams['dropout_1']))
    model.add(LSTM(output_dim=round(hpparams['output_dim_2']),
                   batch_input_shape=(BATCH_SIZE, MAXLEN, len(CHARS)), return_sequences=True))
    model.add(Dropout(hpparams['dropout_2']))
    model.add(LSTM(output_dim=round(hpparams['output_dim_3']),
                   batch_input_shape=(BATCH_SIZE, MAXLEN, len(CHARS)), return_sequences=False))
    model.add(Dense(output_dim=round(hpparams['output_dim_4'])))
    model.add(Dense(output_dim=len(CHARS)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    sent, next_chr = get_sentences()
    sent_val, next_chr_val = get_sentences_val()
    X, y = vectorization(sent, next_chr)
    X_val, y_val = vectorization(sent_val, next_chr_val)

    for epoch in range(hpparams['epochs']):
        print('Training epoch %s of %s' % (epoch, hpparams['epochs']))
        hist = model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1,
                         shuffle=False,
                         validation_data=(X_val, y_val),
                         verbose=1)
        all_results.append(hist.history['val_loss'])
        all_results_train.append(hist.history['loss'])

    return min(all_results), min(all_results_train)


space4dt = {
   'output_dim_1': hp.choice('output_dim_1', (64, 128, 256, 512)),
   'output_dim_2': hp.choice('output_dim_2', (64, 128, 256, 512)),
   'output_dim_3': hp.choice('output_dim_3', (64, 128, 256, 512)),
   'output_dim_4': hp.choice('output_dim_4', (64, 128, 256, 512)),
   'dropout_1': hp.choice('dropout_1', (0, 0.1, 0.2, 0.3, 0.4)),
   'dropout_2': hp.choice('dropout_2', (0, 0.1, 0.2, 0.3, 0.4)),
   'epochs': hp.choice('epochs', (1, 5, 10, 20, 30)),
}


def f(params):
    global val_loss_, loss_, params_, counter
    # val_loss, loss = hyperopt_search_generator(params)
    val_loss, loss = hyperopt_search(params)
    val_loss_.append(val_loss)
    loss_.append(loss)
    params_.append(params_)

    counter += 1
    print_params = {}
    print(counter, round(val_loss, 4), ' ', round(loss, 4), ' ', print_params)
    return {'loss': val_loss, 'status': STATUS_OK}


trials = Trials()
val_loss_, loss_, params_ = [], [], []
counter = 0
best_params = pd.DataFrame()
best = fmin(f, space4dt, algo=tpe.suggest, max_evals=5, trials=None, verbose=1)
print('best:')
print(best)

print('Candidates:')
best_params['val_loss'] = val_loss_
best_params['loss'] = loss_
best_params['params'] = params_

best_params.sort_values(by=['loss', 'std'], inplace=True)
print(best_params)
best_params.to_csv('data/best_params_stratifiedKFold.csv', index=False)


