from datamuse import datamuse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


api = datamuse.Datamuse()

def dm_to_df(datamuse_response):
    """Converts the json response of the datamuse API into a DataFrame
    :datamuse_response
        [{'word': 'foo', 'score': 100}, {'word': 'bar', 'score': 120}]
    """
    reformatted = {
        'word': [response['word'] for response in datamuse_response],
        'score': [response['score'] for response in datamuse_response]
    }
    return pd.DataFrame.from_dict(reformatted)

def getSimilar(word, arg, count):
    #select api call based on argument
    if arg == 'sl':
        response = api.words(sl=word,max=count)
    if arg == 'sp':
        response = api.words(sp=word,max=count)
    if arg == 'ml':
        response = api.words(ml=word,max=count)
    if arg == 'rel_trg':
        response = api.words(rel_trg=word,max=count)
    #make response into a set to return
    return set(dm_to_df(response)['word'])

def jaccardSim(set1, set2):
    inter = set1.intersection(set2)
    uni = set1.union(set2)

    #in case neither got hits
    try:
        result = float(len(inter))/float(len(uni))
    except ZeroDivisionError:
        return 0
    return result

def getPearsons(df_col1, df_col2):
    #arguments are pandas columns of calculated similarities
    arr1 = df_col1.to_numpy()
    arr2 = df_col2.to_numpy()
    return pearsonr(arr1, arr2).statistic

def dmPairSim(word1, word2, call, count):
    #returns jaccard similarity of words based on word sets from datamuse API call responses
    w1_set = getSimilar(word1, call, count)
    w2_set = getSimilar(word2, call, count)
    return jaccardSim(w1_set, w2_set)

def runFileApiCall(file, call, n):
    df = pd.read_csv(file,sep=';', names=['W1','W2','human_sim'])
    df["sim"] = 0
    for i,row in df.iterrows():
        row = row.copy()
        w1 = row["W1"]
        w2 = row["W2"]
        df.loc[i,"sim"] = dmPairSim(w1,w2,call,n)
    return getPearsons(df["human_sim"], df["sim"])

def test():    
    result = runFileApiCall('datasets/mc.csv', 'rel_trg',100)
    print(result)

def test2():    
    df = pd.read_csv("datasets/ssts-131.csv",sep=';',names=['S1','S2','human_sim'])
    df["sim"] = float (0)
    for i,row in df.iterrows():
        row = row.copy()
        sentence = row['S1']
        df.loc[i,"sim"] = jaccardSim()
    print(df)
    print(getPearsons(df["human_sim"],df["sim"]))

test()