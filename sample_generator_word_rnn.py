import numpy as np
import random
import re
import sys
from keras.models import model_from_json, Sequential

# global params
MAXLEN = 20
STEP = 1
BATCH_SIZE = 1000


def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    if sum(a) > 1.0:
        a *= 1 - (sum(a) - 1)
        if sum(a) > 1.0:
            a *= 0.99999
    return np.argmax(np.random.multinomial(1, a, 1))


def get_sample(model, temperatures):  # [0.2, 0.5, 1.0]
    raw_text = open('data/Lev_Tolstoy_all.txt', encoding="utf-8").read()
    raw_text = raw_text.lower()
    raw_text_ru = re.sub("[^а-я, .]", "", raw_text)
    chars = sorted(list(set(raw_text_ru)))
    print('Length chars', len(chars))
    print(chars)

    for T in temperatures:
        print("------------Temperature", T)
        generated = ''
        # sentence = 'и дело не в исполнен'
        sentence = 'полковой командир, п'
        generated += sentence
        print("Generating with seed: " + sentence)
        print('')

        for i in range(2000):
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

path = 'models/'

json_file = open(
    path + 'current_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
trained_model = model_from_json(loaded_model_json)
trained_model.load_weights(
    path + 'weights_ep_19_loss_1.090_val_loss_1.245.hdf5')
trained_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

get_sample(trained_model, [0.2, 0.3, 0.4, 0.5, 1.0])




