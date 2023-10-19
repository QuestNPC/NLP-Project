from datamuse import datamuse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import os
import re
import contractions
import dataAnalysis
import nltk
from nltk.corpus import wordnet
from antonyms import word_antonym_replacer

from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity

api = datamuse.Datamuse()
rep_antonym = word_antonym_replacer()

def removeStopwords(sentence):
    stopwords = list(set(nltk.corpus.stopwords.words('english')))
    stopwords.extend(['could', 'would']) #for some reason not included
    stopwords.remove('not') #leave not for antonyms
    x = sentence.split()
    filtered_sentence = [w for w in x if w.isalpha() and w not in stopwords]
    #with no lower case conversion
    return filtered_sentence

def preprocess(sentence):
    #gets rid of relevant special chars 
    sentence = str(sentence).lower().replace('!', '').replace('?', '').replace('.', '').replace(',', '')
    sentence = contractions.fix(sentence)
    sentence = removeStopwords(sentence)
    sentence = rep_antonym.replace_negations(sentence)
    if 'not' in sentence:
        sentence.remove('not')
    return sentence


def task8(method):
    #method on string joka valitsee millä tavalla similarity otetaan

    #loopissa rivetiiäin lausetia läpi
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df["sim"] = float (0)

    concept = ConceptSimilarity(Taxonomy(DBpediaDataTransform()),'models/dbpedia_type_ic.txt')

    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]
        s1_tokens = preprocess(s1)
        s2_tokens = preprocess(s2)
        
        concepts1 = []
        for token in s1_tokens:
            concepts1.append(concept.name2concept(token))
        similarities1 = []

        concepts2 = []
        for token in s2_tokens:
            concepts2.append(concept.name2concept(token))
        similarities2 = []
        
        for c1 in concepts1:
            max = 0
            for c2 in concepts2:
                sim = concept.similarity(c1, c2, method)
                if sim > max:
                    max =  sim
            similarities1.append(max)
        sim1 = similarities1.mean()

        for c2 in concepts2:
            max = 0
            for c1 in concepts1:
                sim = concept.similarity(c2, c1, method)
                if sim > max:
                    max =  sim
            similarities2.append(max)
        sim2 = similarities2.mean()
        sim = float((sim1+sim2)/2)
        df.loc[i,"sim"] = sim
        dataAnalysis.getPearsons(df["human_sim"], df["sim"])

def test5():
    df1 = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df = df1[['S1','S2']].copy()
    df['S1_tokens'] = None
    df['S2_tokens'] = None
    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]
        s1 = preprocess(s1)
        s2 = preprocess(s2)
        df.at[i,"S1_tokens"] = s1
        df.at[i,"S2_tokens"] = s2
    df.to_csv('preprocess_test.csv', index=False, sep=';')
def testing():
    print(rep_antonym.replace('like'))

if __name__ == "__main__":
    test5()