import pandas as pd
import numpy as np
import dataAnalysis
import preprocessing
import torchtext
import gensim.downloader
from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity
from scipy.spatial.distance import cosine
import fasttext
import fasttext.util
from sematch.semantic.similarity import YagoTypeSimilarity

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
            v1.append(np.zeros(300))
    vec_s1 = np.mean(v1, axis=0)

    #vec_s2 = np.mean([model.get_vector(x) for x in s2.split()], axis=0)
    v2 = []
    for w in s2.split():
        try:
            v2.append(model.get_vector(w))
        except KeyError:
            v2.append(np.zeros(300))
    vec_s2 = np.mean(v2, axis=0)
    cos_sim = 1 - cosine(vec_s1,vec_s2)
    return cos_sim

def gloveAnalysis(contractions=True):
    #GloVe model
    model = torchtext.vocab.GloVe(name='6B', dim=300)
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df["sim"] = float (0)
    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]
        s1 = preprocessing.preprocessAmerican(s1, contractions)
        s2 = preprocessing.preprocessAmerican(s2, contractions)
        df.loc[i,"sim"] = get_glove_sim(s1, s2, model)

    if contractions:
        fname = 'results/Glove_contractions_ssts.csv'
    else:
        fname = 'results/Glove_ssts.csv'
    df.to_csv(fname, index=False, header=True)

    print('Correlation for: ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def w2vAnalysis(contractions=True):
    model = gensim.downloader.load('word2vec-google-news-300')
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df["sim"] = float (0)
    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]
        s1 = preprocessing.preprocessAmerican(s1, contractions)
        s2 = preprocessing.preprocessAmerican(s2, contractions)
        df.loc[i,"sim"] = get_w2v_sim(s1, s2, model)
    if contractions:
        fname = 'results/w2v_contractions_ssts.csv'
    else:
        fname = 'results/w2v_ssts.csv'
    df.to_csv(fname, index=False, header=True)
    print('Correlation for: ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def ftAnalysis(contractions=True):
    #contractions is either True of False
    fasttext.util.download_model('en', if_exists='ignore')
    model = fasttext.load_model('cc.en.300.bin')
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df["sim"] = float (0)
    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]
        s1 = preprocessing.preprocessAmerican(s1,contractions)
        s2 = preprocessing.preprocessAmerican(s2,contractions)
        df.loc[i,"sim"] = get_ft_sim(s1, s2, model)

    if contractions:
        fname = 'results/ft_contractions_ssts.csv'
    else:
        fname = 'results/ft_ssts.csv'
    df.to_csv(fname, index=False, header=True)

    print('Correlation for: ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))


def BDpedia(exclude=False):
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

            sim = bdpediaSim(s1_tokens, s2_tokens, concept,  method, exclude)
            df.loc[i,"sim"] = sim
        if exclude:
            fname = 'results/DBpedia_exlude_' + method + '.csv'
        else:
            fname = 'results/DBpedia_' + method + '.csv'
        df.to_csv(fname, index=False, header=True)

        print('Correlation for ', method, ': ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def bdpediaSim(s1_tokens, s2_tokens, concept, method = 'path', exclude = True):
    sim = 0
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
            similarities1.append(bdMax(method, concept, concepts2, c1))
        sim1 = np.average(similarities1)

        for c2 in concepts2:
            similarities2.append(bdMax(method, concept, concepts1, c2))
        sim2 = np.average(similarities2)
        sim = float((sim1+sim2)/2)
    return sim

def bdMax(method, concept, concepts2, c1):
    if c1 == []:
        return 0
    max = 0
    for c2 in concepts2:
        if not c2 == []:
            sim = concept.similarity(c1, c2, method)
            if sim > max:
                max =  sim
    return max

def yago(exclude=False):
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

            sim = yagoSim(s1_tokens, s2_tokens, yago_sim, method, exclude)
            df.loc[i,"sim"] = sim
        if exclude:
            fname = 'results/yago_exlude_' + method + '.csv'
        else:
            fname = 'results/yago_' + method + '.csv'
        df.to_csv(fname, index=False, header=True)

        print('Correlation for ', method, ': ', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def yagoSim( t1, t2, yago_sim, method='path', exclude=True):

    sim = 0
            
    concepts1 = []
    for token in t1:
        concepts1.append(yago_sim.word2yago(token))
    similarities1 = []

    concepts2 = []
    for token in t2:
        concepts2.append(yago_sim.word2yago(token))
    similarities2 = []

            #comment these out to get results where tokens without concept get simimlarity 0
            
    if exclude:
        concepts1 = [c for c in concepts1 if not c == []]
        concepts2 = [c for c in concepts2 if not c == []]

    if not (concepts2 == [] or concepts1 == []): #just making sure no issues happen if sentences get no matches
                #stupid tree #1
        for c1 in concepts1:
            similarities1.append(yagomax(yago_sim, method, concepts2, c1))
        sim1 = np.average(similarities1)

                #stupid tree #2
        for c2 in concepts2:
            similarities2.append(yagomax(yago_sim, method, concepts1, c2))
        sim2 = np.average(similarities2)

        sim = float((sim1+sim2)/2)
    return sim

def yagomax(yagoo, method, concept_list, conc):
    if conc == []:
        return 0
    max = 0
    for c_1 in conc:
        for c2 in concept_list:
            if not c2 == []:
                for c_2 in c2:
                    sim = yagoo.yago_similarity(c_1, c_2, method)
                    if sim > max:
                        max =  sim
    return max

def testing():
    gloveAnalysis(True)
    gloveAnalysis(False)
    w2vAnalysis()
    w2vAnalysis(False)
    ftAnalysis(True)
    ftAnalysis(False)


if __name__ == "__main__":
    #testing()
    #BDpedia(True)
    #BDpedia(False)
    #yago(True)
    #yago(False)
    concept = ConceptSimilarity(Taxonomy(DBpediaDataTransform()),'models/dbpedia_type_ic.txt')
    print(bdpediaSim(['actor', 'film'], ['actor','train'], concept))