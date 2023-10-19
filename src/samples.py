#contains sampple code to start work with

#datamuse sample imports
from datamuse import datamuse
from datamuse import scripts

#lab2 imports
import nltk
import numpy as np
from numpy.linalg import norm
from nltk.corpus import genesis
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer
from fuzzywuzzy import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.spatial.distance import cosine

import spacy

from nltk.corpus import brown
from gensim.models import Word2Vec

import pandas as pd

import torch
import torchtext

import re

import fasttext
import fasttext.util

#some code from lab2
def removeStopwords(x):
    stopwords = list(set(nltk.corpus.stopwords.words('english')))
    filtered_sentence = [w.lower() for w in x if w.isalpha() and w.lower() not in stopwords]
    #with no lower case conversion
    return filtered_sentence


stemmer = nltk.PorterStemmer()

def stem_tokens(x):
    return [stemmer.stem(w) for w in x]


def get_ft_sim(s1, s2, ftmodel):
    #no preprocessing as instructed, just splitting to get each word as they appear in sentence
    vec_s1 = np.mean([ftmodel[x] for x in s1.split()], axis=0)
    vec_s2 = np.mean([ftmodel[x] for x in s2.split()], axis=0)

    cos_sim = 1 - cosine(vec_s1,vec_s2)
    return cos_sim

#example word2vector model using brown corpus
w2v_model = Word2Vec(brown.sents())
#GloVe model
glove = torchtext.vocab.GloVe(name='6B', dim=50)

def get_glove_sim(s1, s2, model):
    #not having words in lowercase and stemmed, fucks with the vectors resulting some to be 0s, but it is what was instructed...
    vec_s1 = np.mean([np.array(model[x]) for x in s1.split()], axis=0)
    vec_s2 = np.mean([np.array(model[x]) for x in s2.split()], axis=0)
    
    cos_sim = 1 - cosine(vec_s1,vec_s2)
    return cos_sim

def get_w2v_sim(s1, s2, model):
    vec_s1 = np.mean([model.wv.get_vector(x) for x in s1.split()], axis=0)
    vec_s2 = np.mean([model.wv.get_vector(x) for x in s2.split()], axis=0)
    
    cos_sim = 1 - cosine(vec_s1,vec_s2)
    return cos_sim

#datamuse examples
api = datamuse.Datamuse()
orange_rhymes = api.words(rel_rhy='orange', max=5)

orange_near_rhymes = api.words(rel_nry='orange', max=5)

foo_complete = api.suggest(s='foo', max=10)

foo_df = scripts.dm_to_df(foo_complete)

def Preprocessing(x):
    def removeEmails(x):
        return re.sub(r'([a-z0-9+._-]+@[a-z0-9+._-]+\.[a-z0-9+_-]+)',"", x)

    def removeUrls(x):
        return re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '' , x)

    def removeRt(x):
        return re.sub(r'\brt\b', '', x).strip()

    def removeSpecialChars(x):
        x = re.sub(r'[^\w ]+', "", x)
        x = ' '.join(x.split())
        return x

    def removeAccentedChars(x):
        x = unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return x

    def removeStopwords(x):
        return ' '.join([t for t in x.split() if t not in stopwords])

    def removeDupsChar(x):
        x = re.sub("(.)\\1{2,}", "\\1", x)
        return x

    def replacing_single_char(x):
        replacements  = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 
                   5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine', 
                   'n':'north','s':'south','e':'east','w':'west'}
        def replace(t):
            if t in replacements:
                return replacements[t]
            else:
                return None
        
        return ' '.join(list(filter(lambda x: x is not None, [replace(t) if len(t) == 1 else t for t in x.split()])))

    x = str(x).lower().replace('\\', '').replace('_', ' ')
#     x = removeStopwords(x)
    x = removeEmails(x)
    x = removeUrls(x)
    x = removeRt(x)
    x = contractions.fix(x)  # update contractions
    x = removeDupsChar(x)
    x = removeAccentedChars(x)
    x = removeSpecialChars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    x = re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", x)  # punctuations, special chars
    x = re.sub(r'\s+', ' ', x) # Extra spaces
#     x = replacing_single_char(x)
    return x