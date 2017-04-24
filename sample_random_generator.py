import numpy as np
import random
import re
import sys
from keras.models import model_from_json, Sequential

# global params
MAXLEN = 30
STEP = 1
BATCH_SIZE = 500


def generate_text(model, length):
    raw_text = open('data/Lev_Tolstoy_all.txt', encoding="utf-8").read()
    raw_text = raw_text.lower()
    raw_text_ru = re.sub("[^а-я, .]", "", raw_text)
    chars = sorted(list(set(raw_text_ru)))
    VOCAB_SIZE = len(chars)

    ix_to_char = {ix: char for ix, char in enumerate(chars)}

    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i + 1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ''.join(y_char)


def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    if sum(a) > 1.0:
        a *= 1 - (sum(a) - 1)
        if sum(a) > 1.0:
            a *= 0.9999
    return np.argmax(np.random.multinomial(1, a, 1))


def get_sample(model, temperatures):  # [0.2, 0.5, 1.0]
    raw_text = open('data/dost_best.txt', encoding="utf-8").read()
    raw_text = raw_text.lower()
    raw_text_ru = re.sub("[^а-я, .]", "", raw_text)
    chars = sorted(list(set(raw_text_ru)))
    print('Length chars', len(chars))
    print(chars)

    start_index = random.randint(0, len(raw_text_ru) - MAXLEN - 1)
    for T in temperatures:
        print("------------Temperature", T)
        generated = ''
        # sentence = raw_text_ru[start_index:start_index + MAXLEN]
        sentence = 'белая плотная шапка пены, боль'
        # sentence = ' бормотал он в смущении, — я так и думал'
        generated += sentence
        print("Generating with seed: " + sentence)
        print('')

        for i in range(1000):
            char_to_int = dict((c, i) for i, c in enumerate(chars))
            int_to_char = dict((i, c) for i, c in enumerate(chars))

            seed = np.zeros((BATCH_SIZE, MAXLEN, len(chars)))
            for t, char in enumerate(sentence):
                seed[0, t, char_to_int[char]] = 1

            predictions = model.predict(seed, batch_size=BATCH_SIZE, verbose=2)[0]
            next_index = sample(predictions, T)
            next_char = int_to_char[next_index]

            sys.stdout.write(next_char)
            sys.stdout.flush()

            generated += next_char
            sentence = sentence[1:] + next_char
        print()

path = 'models_dostoevsky/'

json_file = open(
    path + 'current_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
trained_model = model_from_json(loaded_model_json)

trained_model.load_weights(path + 'weights_ep_12_loss_1.216_val_loss_1.316.hdf5')
trained_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

get_sample(trained_model, [0.4])
# generate_text(trained_model, 100)
