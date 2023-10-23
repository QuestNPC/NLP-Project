from datamuse import datamuse
import pandas as pd
import numpy as np
import dataAnalysis
from nltk.corpus import wordnet
import preprocessing
import torch
import torchtext
import gensim.downloader
from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine
import fasttext
from breame.spelling import british_spelling_exists, get_american_spelling
import fasttext.util
from sematch.semantic.similarity import YagoTypeSimilarity


api = datamuse.Datamuse()

#example word2vector model using brown corpus

def task5():
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df["sim"] = float (0)

    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]
        s1_tokens = preprocessing.preprocess(s1)
        s2_tokens = preprocessing.preprocess(s2)
        
        s1_sets = []
        s2_sets = []
        for token in s1_tokens:
            #get set of words with best method
            #append the s1_sets list
            continue
        #get intersection of all sets in s1_sets and add those to s1_tokens

        for token in s2_tokens:
            #get set of words with best method
            #append the s2_sets list
            continue
        #get intersection of all sets in s2_sets and add those to s2_tokens
        
        #jaccardsim the tokens
        #df.loc[i,"sim"] = jaccardsim
    fname = 'results/Dmuse_ssts.csv'
    df.to_csv(fname, index=False, header=True)

    print('Correlation: ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def get_ft_sim(s1, s2, ftmodel):
    vec_s1 = np.mean([ftmodel[x] for x in s1.split()], axis=0)
    vec_s2 = np.mean([ftmodel[x] for x in s2.split()], axis=0)

    cos_sim = 1 - cosine(vec_s1,vec_s2)
    return cos_sim

def get_glove_sim(s1, s2, model):
    vec_s1 = np.mean([np.array(model[x]) for x in s1.split()], axis=0)
    vec_s2 = np.mean([np.array(model[x]) for x in s2.split()], axis=0)
    
    cos_sim = 1 - cosine(vec_s1,vec_s2)
    return cos_sim

def get_w2v_sim(s1, s2, model):

    #vec_s1 = np.mean([model.get_vector(x) for x in s1.split()], axis=0)
    v1 = []
    for w in s1.split():
        try:
            v1.append(model.get_vector(w))
        except KeyError:
            if british_spelling_exists(w):
                v1.append(model.get_vector(get_american_spelling(w)))
    vec_s1 = np.mean(v1, axis=0)

    #vec_s2 = np.mean([model.get_vector(x) for x in s2.split()], axis=0)
    v2 = []
    for w in s2.split():
        try:
            v2.append(model.get_vector(w))
        except KeyError:
            if british_spelling_exists(w):
                v2.append(model.get_vector(get_american_spelling(w)))
    vec_s2 = np.mean(v2, axis=0)
    cos_sim = 1 - cosine(vec_s1,vec_s2)
    return cos_sim

def gloveAnalysis():
    #GloVe model
    model = torchtext.vocab.GloVe(name='6B', dim=50)
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df["sim"] = float (0)
    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]
        s1 = preprocessing.preprocess2(s1)
        s2 = preprocessing.preprocess2(s2)
        df.loc[i,"sim"] = get_glove_sim(s1, s2, model)

    fname = 'results/Glove_ssts.csv'
    df.to_csv(fname, index=False, header=True)

    print('Correlation for: ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def w2vAnalysis():
    model = gensim.downloader.load('word2vec-google-news-300')
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df["sim"] = float (0)
    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]
        s1 = preprocessing.preprocess2(s1)
        s2 = preprocessing.preprocess2(s2)
        df.loc[i,"sim"] = get_w2v_sim(s1, s2, model)

    fname = 'results/w2v_ssts.csv'
    df.to_csv(fname, index=False, header=True)

    print('Correlation for: ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def ftAnalysis():
    fasttext.util.download_model('en', if_exists='ignore')
    model = fasttext.load_model('cc.en.300.bin')
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df["sim"] = float (0)
    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]
        s1 = preprocessing.preprocess2(s1)
        s2 = preprocessing.preprocess2(s2)
        df.loc[i,"sim"] = get_ft_sim(s1, s2, model)

    fname = 'results/ft_ssts.csv'
    df.to_csv(fname, index=False, header=True)

    print('Correlation for: ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))


def BDpedia(exclude):
    methods = ['path', 'wup','li'] #'lin','jcn', 'res','wpath' ends with certificate expired error
    for method in methods:
        df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
        df["sim"] = float (0)

        concept = ConceptSimilarity(Taxonomy(DBpediaDataTransform()),'models/dbpedia_type_ic.txt')

        for i,row in df.iterrows():
            row = row.copy()
            s1 = row["S1"]
            s2 = row["S2"]
            s1_tokens = preprocessing.preprocess(s1)
            s2_tokens = preprocessing.preprocess(s2)
            
            concepts1 = []
            for token in s1_tokens:
                concepts1.append(concept.name2concept(token))
            similarities1 = []

            concepts2 = []
            for token in s2_tokens:
                concepts2.append(concept.name2concept(token))
            similarities2 = []

            if exclude:
                concepts1 = [c for c in concepts1 if not c == []]
                concepts2 = [c for c in concepts2 if not c == []]

            if not (concepts2 == [] or concepts1 == []):
                for c1 in concepts1:
                    max = 0
                    if not c1 == []:
                        for c2 in concepts2:
                            if not c2 == []:
                                sim = concept.similarity(c1, c2, method)
                                if sim > max:
                                    max =  sim
                    similarities1.append(max)
                sim1 = np.average(similarities1)

                for c2 in concepts2:
                    max = 0
                    if not c2 == []:
                        for c1 in concepts1:
                            if not c1 == []:
                                sim = concept.similarity(c2, c1, method)
                                if sim > max:
                                    max =  sim
                    similarities2.append(max)
                sim2 = np.average(similarities2)
                sim = float((sim1+sim2)/2)
                df.loc[i,"sim"] = sim
        if exclude:
            fname = 'results/DBpedia_exlude_' + method + '.csv'
        else:
            fname = 'results/DBpedia_' + method + '.csv'
        df.to_csv(fname, index=False, header=True)

        print('Correlation for ', method, ': ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def yago(exclude):
    yago_sim = YagoTypeSimilarity()
    methods = ['path', 'wup','li'] #'lin','jcn', 'res','wpath' ends with certificate expired error
    for method in methods:
        df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
        df["sim"] = float (0)

        for i,row in df.iterrows():
            row = row.copy()
            s1 = row["S1"]
            s2 = row["S2"]
            s1_tokens = preprocessing.preprocess(s1)
            s2_tokens = preprocessing.preprocess(s2)
            
            concepts1 = []
            for token in s1_tokens:
                concepts1.append(yago_sim.word2yago(token))
            similarities1 = []

            concepts2 = []
            for token in s2_tokens:
                concepts2.append(yago_sim.word2yago(token))
            similarities2 = []

            #comment these out to get results where tokens without concept get simimlarity 0
            
            if exclude:
                concepts1 = [c for c in concepts1 if not c == []]
                concepts2 = [c for c in concepts2 if not c == []]

            if not (concepts2 == [] or concepts1 == []): #just making sure no issues happen if sentences get no matches
                #stupid tree #1
                for c1 in concepts1:
                    max = 0
                    if not c1 == []:
                        for c_1 in c1:
                            for c2 in concepts2:
                                if not c2 == []:
                                    for c_2 in c2:
                                        sim = yago_sim.yago_similarity(c_1, c_2, method)
                                        if sim > max:
                                            max =  sim
                    similarities1.append(max)
                sim1 = np.average(similarities1)

                #stupid tree #2
                for c2 in concepts2:
                    max = 0
                    if not c2 == []:
                        for c_2 in c2:
                            for c1 in concepts1:
                                if not c1 == []:
                                    for c_1 in c1:
                                        sim = yago_sim.yago_similarity(c_2, c_1, method)
                                        if sim > max:
                                            max =  sim
                    similarities2.append(max)
                sim2 = np.average(similarities2)

                sim = float((sim1+sim2)/2)
                df.loc[i,"sim"] = sim
        if exclude:
            fname = 'results/yago_exlude_' + method + '.csv'
        else:
            fname = 'results/yago_' + method + '.csv'
        df.to_csv(fname, index=False, header=True)

        print('Correlation for ', method, ': ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def testing():
    #gloveAnalysis(glove)
    w2vAnalysis()
    #ftAnalysis(ft_en_model)
    pass

if __name__ == "__main__":
    BDpedia(True)
    BDpedia(False)
