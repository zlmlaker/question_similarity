#coding:utf-8
import re
import csv
import codecs
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from Util import *

import sys
import numpy as np
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
assert len(sequences_1) == len(sequences_2)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))


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

seq1_vec = []
seq2_vec = []
sum_vec = np.zeros([EMBEDDING_DIM])
for idx,seq in enumerate(sequences_1):
    for i in xrange(EMBEDDING_DIM):
        sum_vec[i] = 0
    for word in seq:
        if word <= nb_words:
            sum_vec +=  embedding_matrix[int(word)]
    seq1_vec.append(normalize(sum_vec))
    if idx%1000 == 0:
        print ("processed %d seq1" %idx)

for idx,seq in enumerate(sequences_2):
    for i in xrange(EMBEDDING_DIM):
        sum_vec[i] = 0
    for word in seq:
        if word <= nb_words:
            sum_vec += embedding_matrix[int(word)]
    seq2_vec.append(normalize(sum_vec))
    if idx % 1000 == 0:
        print ("processed %d seq2" % idx)

sim = []
cnt = 0
for v1, v2 in zip(seq1_vec, seq2_vec):
    cnt += 1
    sim.append(cal_sim(v1,v2))
    if cnt % 1000 == 0:
        print ("processed %d cal sim "% cnt)
sim = np.array(sim)

for i in xrange(5):
    print (sim[i])
    print '\n'

print "determin thresh by train data"
best_thresh = 0.0
best_f1 = 0.0
for iter in xrange(0,50):
    thresh = iter * (0.1 / 50) + 0.9
    train_data = sim[0:int(len(sim) *(1 - VALIDATION_SPLIT)) ]
    train_label = labels[0:int(len(sim) *(1 - VALIDATION_SPLIT))]
    pred = np.zeros_like(train_data)
    pred[train_data > thresh] = 1
    recall = recall_np(train_label, pred)
    precision = precision_np(train_label,pred)
    f1 = f1_np(precision, recall)
    print "thresh:%f precision:%f recall:%f f1:%f" %(thresh, precision,recall,f1)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

print "best_thresh:%f" % best_thresh
print "eval on test data"
test_data = sim[ int(len(sim)*(1 - VALIDATION_SPLIT)) :]
test_labels = labels[int(len(sim) *(1 - VALIDATION_SPLIT)):]
pred = np.zeros_like(test_data)
pred[test_data > best_thresh] = 1
recall = recall_np(test_labels, pred)
precision = precision_np(test_labels, pred)
f1 = f1_np(precision, recall)
print "precision:%f recall:%f f1:%f" % (precision, recall, f1)






