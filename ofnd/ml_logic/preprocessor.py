from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import string
import unidecode
import nltk
import pandas as pd
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from ofnd.ml_logic.params import TARGET_COLUMN, TRUE_LOCAL_PATH, FAKE_LOCAL_PATH
from ofnd.ml_logic.params import MODEL_TYPE
import json, re
from tqdm import tqdm_notebook
from uuid import uuid4

#tf modules
import tensorflow as tf
from tensorflow import keras
import gensim.downloader as api
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras import layers, Sequential, optimizers, metrics, models

## Torch Modules
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup

stop_words = set(stopwords.words('english')) ## define stopwords




def get_data(TRUE_LOCAL_PATH, FAKE_LOCAL_PATH):
    fake_df = pd.read_csv(FAKE_LOCAL_PATH)
    true_df = pd.read_csv(TRUE_LOCAL_PATH)
    TRUE_LOCAL_PATH['True'] = 1
    FAKE_LOCAL_PATH['True'] = 0
    articles_df = pd.concat([true_df, fake_df])

    return articles_df


def preparation(df):

    articles_df['title_text'] = articles_df['title'] + articles_df['text']
    articles_df = articles_df.drop(columns= ['title', 'text', 'date'])

    return articles_df

def removing_words(df):
  list_of_words = ['Donald', 'Trump', 'Dont', 'donald', 'trump', 'dont']

  pat = r'\b(?:{})\b'.format('|'.join(list_of_words))

  df['news'] = df['news'].str.replace(pat, '')

  return df

def clean(sentence):

    if MODEL_TYPE == 'tensorflow':
        word2vec_transfer = api.load("glove-wiki-gigaword-50")

        X_embed = embedding(word2vec_transfer, sentence)

        X_pad = pad_sequences(X_embed, dtype='float32', padding='post', maxlen=100)

        return X_pad

    if MODEL_TYPE == 'roberta':
        roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        sentences = sentence['news'].values

        roberta_input_ids = []
        roberta_attention_masks = []
        sentence_ids = []
        counter = 0

        for sent in sentences:
            roberta_encoded_dict = roberta_tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = False, # Add '[CLS]' and '[SEP]'
                        max_length = 120,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt')     # Return pytorch tensors.

            roberta_input_ids.append(roberta_encoded_dict['input_ids'])

            roberta_attention_masks.append(roberta_encoded_dict['attention_mask'])

        roberta_input_ids = torch.cat(roberta_input_ids, dim=0)
        roberta_attention_masks = torch.cat(roberta_attention_masks, dim=0)

        batch_size = 32

        prediction_data = TensorDataset(roberta_input_ids, roberta_attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        return prediction_dataloader

    if MODEL_TYPE == 'ml':
        # Basic cleaning
        sentence = sentence.strip() ## remove whitespaces
        sentence = sentence.lower() ## lowercase
        sentence = ''.join(char for char in sentence if not char.isdigit()) ## remove numbers

        # Advanced cleaning
        for punctuation in (string.punctuation + "…”"):
            sentence = sentence.replace(punctuation, ' ') ## remove punctuation

        unaccented_string = unidecode.unidecode(sentence) # remove accents

        tokenized_sentence = word_tokenize(unaccented_string) ## tokenize
        # stop_words = set(stopwords.words('english')) ## define stopwords

        tokenized_sentence_cleaned = [ ## remove stopwords
            w for w in tokenized_sentence if not w in stop_words
                ]

        lemmatized = [
            WordNetLemmatizer().lemmatize(word, pos = "v")
            for word in tokenized_sentence_cleaned
        ]

        cleaned_sentence = ' '.join(word for word in lemmatized)

        return cleaned_sentence

def clean_data(X):
    if MODEL_TYPE == 'roberta':
        return clean(X)
    if MODEL_TYPE ==  'tensorflow':
        return clean(X)
    if MODEL_TYPE == 'ml':
        if isinstance(X, pd.Series):
            return X.apply(clean)
        if isinstance(X, pd.DataFrame):
            return X.applymap(clean)


def X_y(df, TARGET_COLUMN):
    X = df.drop([TARGET_COLUMN], axis=1)
    y = df[TARGET_COLUMN]
    return X, y


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y,\
        test_size=0.3, random_state=0, shuffle=True)

    return X_train, X_test, y_train, y_test


def train_vect(X):

    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
    tfidf = tfidf_vectorizer.fit(X)
    return tfidf


def transform_vect(X):

    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
    tfidf = tfidf_vectorizer.transform(X)

    return tfidf


def embed_sentence_with_TF(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec:
            embedded_sentence.append(word2vec[word])

    return np.array(embedded_sentence)

def embedding(word2vec, sentences):
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence_with_TF(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed
