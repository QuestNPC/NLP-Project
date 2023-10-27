import pandas as pd
import dataAnalysis
import preprocessing

from nltk.util import bigrams, trigrams

def setintersection(set_list):
    isect = set_list[0].intersection(set_list[1])
    i = 2
    while i < len(set_list):
        isect = isect.intersection(set_list[i])
        i += 1
    return isect

def task5(mode=0):
    fname = 'results/Dmuse_ssts_'+ str(mode) +'.csv'
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim','std'])
    df["sim"] = float (0)
    for i,row in df.iterrows():
        row = row.copy()
        s1 = row["S1"]
        s2 = row["S2"]

        if  mode == 1:
            #jaccard sim of tokens
            s1_tokens = preprocessing.preprocess(s1)
            s2_tokens = preprocessing.preprocess(s2)
            sim = dataAnalysis.jaccardSim(set(s2_tokens),set(s1_tokens))
        elif mode == 2:
            #querries with tokens and extends tokens with union of response set bigram intersections
            sim = mode2(s1, s2)
        elif mode == 3:
            #querries with tokens and extends tokens with union of response set triigram intersections
            sim = mode3(s1, s2)
        elif mode == 4:
            #querries with bigrams from sentence, extends sentence token with union of response sets
            sim = mode4(s1,s2)
        elif mode == 5:
            #querries with trigrams from sentence, extends sentence token with union of response sets
            sim = mode5(s1,s2)
        elif mode == 6:
            #querries with bigrams from sentence, extends sentence token with intersection of response sets
            sim = mode6(s1, s2)
        elif mode == 7:
            #querries with triigrams from sentence, extends sentence token with intersection of response sets
            sim = mode7(s1, s2)
        elif mode == 8:
            #querry with full sentence
            sim = datamsueFull(s1, s2)
        elif mode == 9:
            #4 wihtout tokens
            sim = mode9(s1,s2)
        elif mode == 10:
            #5 without tokens
            sim = mode10(s1,s2)
        elif mode == 11:
            #6 without tokens
            sim = mode11(s1, s2)
        elif mode == 12:
            #7 without tokens
            sim = mode12(s1, s2)
        elif mode == 13:
            #querries with tokens and extends tokens with union over all responses
            sim = mode13(s1, s2)
        else:
            #querries with tokens and extends tokens with intersection over all responses
            sim = tokensAll(s1, s2)
        df.loc[i,"sim"] = sim
    
    df.to_csv(fname, index=False, header=True)
    
    print('Correlation for mode', mode, ':', dataAnalysis.getPearsons(df["human_sim"], df["sim"]))

def datamsueFull(s1, s2):
    sent1 = preprocessing.preprocess2(s1)
    sent2 = preprocessing.preprocess2(s1)
    s1_tokens = preprocessing.preprocess(s1)
    s2_tokens = preprocessing.preprocess(s2)
    s1_tokens.extend(list(dataAnalysis.bestAPIcallset(sent1, 1000)))
    s2_tokens.extend(list(dataAnalysis.bestAPIcallset(sent2, 1000)))
    sim = dataAnalysis.jaccardSim(set(s2_tokens),set(s1_tokens))
    return sim

def mode2(s1, s2):
    s1_sets = []
    s1_tokens = preprocessing.preprocess(s1)
    s2_tokens = preprocessing.preprocess(s2)
    for token in s1_tokens:
        w_set = dataAnalysis.bestAPIcallset(token, 1000)
        s1_sets.append(w_set)
    s1_isect = setBigramIntersection(s1_sets)

    s2_sets = []
    for token in s2_tokens:
        w_set = dataAnalysis.bestAPIcallset(token, 1000)
        s2_sets.append(w_set)
    s2_isect = setBigramIntersection(s2_sets)

    s1_tokens.extend(list(s1_isect))
    s2_tokens.extend(list(s2_isect))
    sim = dataAnalysis.jaccardSim(set(s2_tokens),set(s1_tokens))
    return sim

def setBigramIntersection(set_list):
    if len(set_list) < 1:
        return set()
    b_sections = []
    i = 1
    #get set pair intersections
    while i < len(set_list):
        b_sections.append(set_list[i-1].intersection(set_list[i]))
        i += 1
    #union of set pair intersections
    additions = set()
    for isection in b_sections:
        additions = additions.union(isection)
    return additions

def mode3(s1, s2):
    s1_tokens = preprocessing.preprocess(s1)
    s2_tokens = preprocessing.preprocess(s2)
    s1_sets = []
    for token in s1_tokens:
        w_set = dataAnalysis.bestAPIcallset(token, 1000)
        s1_sets.append(w_set)
    s1_isect = trigramSetIntersection(s1_sets)

    s2_sets = []
    for token in s2_tokens:
        w_set = dataAnalysis.bestAPIcallset(token, 1000)
        s2_sets.append(w_set)
    s2_isect = trigramSetIntersection(s2_sets)

    s1_tokens.extend(list(s1_isect))
    s2_tokens.extend(list(s2_isect))
    sim = dataAnalysis.jaccardSim(set(s2_tokens),set(s1_tokens))
    return sim

def trigramSetIntersection(set_list):
    if len(set_list) < 2:
        return set()
    b_sections = []
    i = 2
    while i < len(set_list):
        b_sections.append(set_list[i-2].intersection(set_list[i-1].intersection(set_list[i])))
        i += 1
    additions = set()
    for isection in b_sections:
        additions = additions.union(isection)
    return additions

def mode4(s1, s2, included):
    sent1 = preprocessing.preprocess2(s1)
    sent2 = preprocessing.preprocess2(s2)
    s1_tokens = preprocessing.preprocess(s1)
    s2_tokens = preprocessing.preprocess(s2)
    s_1 = sent1.split()
    s_2 = sent2.split()
    bigrams1 = list(bigrams(s_1))
    bigrams2 = list(bigrams(s_2))
    
    s1_sets = []
    for big in bigrams1:
        s1_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000))
    
    s2_sets = []
    for big in bigrams2:
        s2_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000)) 

    additions1 = set()
    for s in s1_sets:
        additions1 = additions1.union(s)
    additions2 = set()
    for s in s2_sets:
        additions2 = additions2.union(s)
    if included:
        s1_tokens.extend(list(additions1))
        s2_tokens.extend(list(additions2))
    else:
        s1_tokens = list(additions1)
        s2_tokens = list(additions2)
    return dataAnalysis.jaccardSim(set(s1_tokens),set(s2_tokens))

def mode5(s1, s2, included):
    
    sent1 = preprocessing.preprocess2(s1)
    sent2 = preprocessing.preprocess2(s2)
    s1_tokens = preprocessing.preprocess(s1)
    s2_tokens = preprocessing.preprocess(s2)
    s_1 = sent1.split()
    s_2 = sent2.split()
    trigrams1 = list(trigrams(s_1))
    trigrams2 = list(trigrams(s_2))
    
    s1_sets = []
    for big in trigrams1:
        s1_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000))
    
    s2_sets = []
    for big in trigrams2:
        s2_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000)) 

    additions1 = set()
    for s in s1_sets:
        additions1 = additions1.union(s)
    additions2 = set()
    for s in s2_sets:
        additions2 = additions2.union(s)
    if included:
        s1_tokens.extend(list(additions1))
        s2_tokens.extend(list(additions2))
    else:
        s1_tokens = list(additions1)
        s2_tokens = list(additions2)
    return dataAnalysis.jaccardSim(set(s2_tokens),set(s1_tokens))

def mode6(s1, s2):
    sent1 = preprocessing.preprocess2(s1)
    sent2 = preprocessing.preprocess2(s2)
    s1_tokens = preprocessing.preprocess(s1)
    s2_tokens = preprocessing.preprocess(s2)
    s_1 = sent1.split()
    s_2 = sent2.split()
    bigrams1 = list(bigrams(s_1))
    bigrams2 = list(bigrams(s_2))
    
    s1_sets = []
    for big in bigrams1:
        s1_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000))
    
    s2_sets = []
    for big in bigrams2:
        s2_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000)) 

    additions1 = setintersection(s1_sets)
    additions2 = setintersection(s2_sets)

    s1_tokens.extend(list(additions1))
    s2_tokens.extend(list(additions2))
    return dataAnalysis.jaccardSim(set(s1_tokens),set(s2_tokens))

def mode7(s1,s2):
    sent1 = preprocessing.preprocess2(s1)
    sent2 = preprocessing.preprocess2(s2)
    s1_tokens = preprocessing.preprocess(s1)
    s2_tokens = preprocessing.preprocess(s2)
    s_1 = sent1.split()
    s_2 = sent2.split()
    trigrams1 = list(trigrams(s_1))
    trigrams2 = list(trigrams(s_2))
    
    s1_sets = []
    for big in trigrams1:
        s1_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000))
    
    s2_sets = []
    for big in trigrams2:
        s2_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000)) 

    additions1 = setintersection(s1_sets)
    additions2 = setintersection(s2_sets)

    s1_tokens.extend(list(additions1))
    s2_tokens.extend(list(additions2))
    return dataAnalysis.jaccardSim(set(s1_tokens),set(s2_tokens))

def mode9(s1, s2):
    sent1 = preprocessing.preprocess2(s1)
    sent2 = preprocessing.preprocess2(s2)
    s_1 = sent1.split()
    s_2 = sent2.split()
    bigrams1 = list(bigrams(s_1))
    bigrams2 = list(bigrams(s_2))
    
    s1_sets = []
    for big in bigrams1:
        s1_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000))
    
    s2_sets = []
    for big in bigrams2:
        s2_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000)) 

    additions1 = set()
    for s in s1_sets:
        additions1 = additions1.union(s)
    additions2 = set()
    for s in s2_sets:
        additions2 = additions2.union(s)
    s1_tokens = list(additions1)
    s2_tokens = list(additions2)
    return dataAnalysis.jaccardSim(set(s1_tokens),set(s2_tokens))

def mode10(s1, s2):
    sent1 = preprocessing.preprocess2(s1)
    sent2 = preprocessing.preprocess2(s2)
    s_1 = sent1.split()
    s_2 = sent2.split()
    trigrams1 = list(trigrams(s_1))
    trigrams2 = list(trigrams(s_2))
    
    s1_sets = []
    for big in trigrams1:
        s1_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000))
    
    s2_sets = []
    for big in trigrams2:
        s2_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000)) 

    additions1 = set()
    for s in s1_sets:
        additions1 = additions1.union(s)
    additions2 = set()
    for s in s2_sets:
        additions2 = additions2.union(s)
    s1_tokens = list(additions1)
    s2_tokens = list(additions2)
    return dataAnalysis.jaccardSim(set(s2_tokens),set(s1_tokens))

def mode11(s1, s2):
    sent1 = preprocessing.preprocess2(s1)
    sent2 = preprocessing.preprocess2(s2)
    s_1 = sent1.split()
    s_2 = sent2.split()
    bigrams1 = list(bigrams(s_1))
    bigrams2 = list(bigrams(s_2))
    
    s1_sets = []
    for big in bigrams1:
        s1_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000))
    
    s2_sets = []
    for big in bigrams2:
        s2_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000)) 

    s1_tokens = list(setintersection(s1_sets))
    s2_tokens = list(setintersection(s2_sets))
    return dataAnalysis.jaccardSim(set(s1_tokens),set(s2_tokens))

def mode12(s1,s2):
    sent1 = preprocessing.preprocess2(s1)
    sent2 = preprocessing.preprocess2(s2)
    s_1 = sent1.split()
    s_2 = sent2.split()
    trigrams1 = list(trigrams(s_1))
    trigrams2 = list(trigrams(s_2))
    
    s1_sets = []
    for big in trigrams1:
        s1_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000))
    
    s2_sets = []
    for big in trigrams2:
        s2_sets.append(dataAnalysis.bestAPIcallset(' '.join(big),1000)) 

    s1_tokens = list(setintersection(s1_sets))
    s2_tokens = list(setintersection(s2_sets))
    return dataAnalysis.jaccardSim(set(s1_tokens),set(s2_tokens))

def mode13(s1, s2):
    s1_tokens = preprocessing.preprocess(s1)
    s2_tokens = preprocessing.preprocess(s2)
    s1_sets = []
    for token in s1_tokens:
        w_set = dataAnalysis.bestAPIcallset(token, 1000)
        s1_sets.append(w_set)

    s2_sets = []
    for token in s2_tokens:
        w_set = dataAnalysis.bestAPIcallset(token, 1000)
        s2_sets.append(w_set)

    additions1 = set()
    for s in s1_sets:
        additions1 = additions1.union(s)
    additions2 = set()
    for s in s2_sets:
        additions2 = additions2.union(s)

    s1_tokens.extend(list(additions1))
    s2_tokens.extend(list(additions2))
    sim = dataAnalysis.jaccardSim(set(s2_tokens),set(s1_tokens))
    return sim

def tokensAll(s1, s2):
    s1_tokens = preprocessing.preprocess(s1)
    s2_tokens = preprocessing.preprocess(s2)
    s1_sets = []
    for token in s1_tokens:
        w_set = dataAnalysis.bestAPIcallset(token, 1000)
        s1_sets.append(w_set)
    s1_isect = setintersection(s1_sets)

    s2_sets = []
    for token in s2_tokens:
        w_set = dataAnalysis.bestAPIcallset(token, 1000)
        s2_sets.append(w_set)
    s2_isect = setintersection(s2_sets)

    s1_tokens.extend(list(s1_isect))
    s2_tokens.extend(list(s2_isect))
    sim = dataAnalysis.jaccardSim(set(s2_tokens),set(s1_tokens))
    return sim

if __name__ == "__main__":
    task5(0)