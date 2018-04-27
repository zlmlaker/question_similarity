#coding:utf-8
import re
import csv
import codecs
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input,Embedding, Dropout, Conv1D, GlobalAveragePooling1D, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from Util import *

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

########################################
## set directories and parameters
########################################
BASE_DIR = 'data/'
EMBEDDING_FILE = BASE_DIR + 'glove.6B.300d.txt'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1


act = 'relu'

########################################
## index word vectors
########################################
print('Indexing word vectors')
embeddings_index = {}
f = open(EMBEDDING_FILE)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors of word2vec' % len(embeddings_index))

########################################
## process texts in datasets
########################################
print('Processing text dataset')


# The function "text_to_wordlist" is from
# https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text)


texts_1 = []
texts_2 = []
labels = []
with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
        texts_1.append(text_to_wordlist(values[3]))
        texts_2.append(text_to_wordlist(values[4]))
        labels.append(int(values[5]))
print('Found %s texts in train.csv' % len(texts_1))


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 )

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)


########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
# np.random.seed(1234)
perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1) * (1 - VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1) * (1 - VALIDATION_SPLIT)):]

data_train_seq1 = data_1[idx_train]
data_train_seq2 = data_2[idx_train]
labels_train = labels[idx_train]

data_val_seq1 = data_1[idx_val]
data_val_seq2 = data_2[idx_val]
labels_val = labels[idx_val]

emb_layer = Embedding(nb_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


# 1D convolutions that can iterate over the word vectors
conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

# Define inputs
seq1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
seq2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

# Run inputs through embedding
emb1 = emb_layer(seq1)
emb2 = emb_layer(seq2)

# Run through CONV + GAP layers
conv1a = conv1(emb1)
glob1a = GlobalAveragePooling1D()(conv1a)
conv1b = conv1(emb2)
glob1b = GlobalAveragePooling1D()(conv1b)

conv2a = conv2(emb1)
glob2a = GlobalAveragePooling1D()(conv2a)
conv2b = conv2(emb2)
glob2b = GlobalAveragePooling1D()(conv2b)

conv3a = conv3(emb1)
glob3a = GlobalAveragePooling1D()(conv3a)
conv3b = conv3(emb2)
glob3b = GlobalAveragePooling1D()(conv3b)

conv4a = conv4(emb1)
glob4a = GlobalAveragePooling1D()(conv4a)
conv4b = conv4(emb2)
glob4b = GlobalAveragePooling1D()(conv4b)

conv5a = conv5(emb1)
glob5a = GlobalAveragePooling1D()(conv5a)
conv5b = conv5(emb2)
glob5b = GlobalAveragePooling1D()(conv5b)

conv6a = conv6(emb1)
glob6a = GlobalAveragePooling1D()(conv6a)
conv6b = conv6(emb2)
glob6b = GlobalAveragePooling1D()(conv6b)

mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

# We take the explicit absolute difference between the two sentences
# Furthermore we take the multiply different entries to get a different measure of equalness
diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 2 * 32,))([mergea, mergeb])
mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 2 * 32,))([mergea, mergeb])


# Merge the Magic and distance features with the difference layer
merge = concatenate([diff, mul])

# The MLP that determines the outcome
x = Dropout(0.2)(merge)
x = BatchNormalization()(x)
x = Dense(300, activation='relu')(x)

x = Dropout(0.2)(x)
x = BatchNormalization()(x)
pred = Dense(1, activation='sigmoid')(x)

# model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
model = Model(inputs=[seq1, seq2], outputs=pred)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[f1,precision, recall])

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
bst_model_path = "cnn1d" + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

hist = model.fit([data_train_seq1,data_train_seq2], labels_train,
                 validation_data=([data_val_seq1, data_val_seq2], labels_val),
                 epochs=200, batch_size=2048, shuffle=True, callbacks=[early_stopping, model_checkpoint])

model.load_weights(bst_model_path)
bst_val_score = min(hist.history['val_loss'])


