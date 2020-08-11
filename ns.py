import tensorflow as tf
# import pandas as pd
import numpy as np
# import re
import tflearn

from tflearn.data_utils import to_categorical

# from nltk.stem.snowball import EnglishStemmer
# from collections import Counter

# vocab_size = 5000

Dataset = open('dataset', 'r')

musical_instruments = []
p_l_g = []

# из dataset достаем отзывы
for line in Dataset:
    if 'musical_instruments' in line:
        musical_instruments.append(line[:line.find('musical_instruments') - 1])
    else:
        p_l_g.append(line[:line.find("patio, lawn and garden") - 1])

Dataset.close()

# создаем список слов

lst_word = []
for review in musical_instruments + p_l_g:
    words = review.lower().split(' ')
    # print(words)
    for word in words:
        if not word in lst_word:
            lst_word.append(word)
        print('процесс идет, стадия 1')


# создаем вектор

def vector_of_zero_and_ones(review):
    vector = []
    words = review.lower().split(' ')
    for word in lst_word:
        if word in words:
            vector.append(1)
        else:
            vector.append(0)

        print('процесс идет, стадия 2')
    return vector


print(vector_of_zero_and_ones('i love my new guitar'))

# тестовые данные
labels = np.append(np.zeros(len(musical_instruments), dtype=np.int_), np.ones(len(p_l_g), dtype=np.int_))
review_vectors = np.zeros((len(musical_instruments) + len(p_l_g),), dtype=np.int_)
reviews = []

for ii, review in enumerate(musical_instruments):
    reviews.append(review)
    review_vectors[ii] = vector_of_zero_and_ones(review)
for ii, review in enumerate(p_l_g):
    reviews.append(review)
    review_vectors[ii + len(musical_instruments)] = vector_of_zero_and_ones(review)
X = review_vectors
y = to_categorical(labels, 2)


# делаем нс

def build_model(learning_rate=0.1):
    tf.reset_default_graph()

    net = tflearn.input_data([None, len(lst_word)])
    net = tflearn.fully_connected(net, 125, activation='ReLU')
    net = tflearn.fully_connected(net, 25, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    regression = tflearn.regression(
        net,
        optimizer='sgd',
        learning_rate=learning_rate,
        loss='categorical_crossentropy')

    model = tflearn.DNN(net)
    return model


model = build_model(learning_rate=0.75)

model.fit(
    X,
    y,
    validation_set=0.1,
    show_metric=True,
    batch_size=128,
    n_epoch=30)


def test_review(review):
    review_vector = vector_of_zero_and_ones(review)
    positive_prob = model.predict([review_vector])[0][1]
    print('Original review: {}'.format(review))
    print('P(positive) = {:.5f}. Result: '.format(positive_prob),
          'musical_instruments' if positive_prob > 0.5 else 'patio, lawn and garden')


review_for_testing = ["i love this new guitar", "I don't think these strings fit my ukulele", "this garden gnome is "
                                                                                              "very scary, "
                                                                                              "I am afraid of my "
                                                                                              "garden", "for my lawn "
                                                                                                        "this is the "
                                                                                                        "same",
                      "sounds great, but not for big concerts", "why the grass is not green, but some kind of yellow",
                      "", ""]
for review in review_for_testing:
    test_review(review)
    print("---------")
