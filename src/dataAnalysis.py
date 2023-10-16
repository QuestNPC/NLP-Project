from datamuse import datamuse
from datamuse import scripts
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


api = datamuse.Datamuse()

def getSimilar(word, arg, count):
    #select api call based on argument
    if arg == 'rel_rhy':
        response = api.words(rel_rhy=word, max=count)   
    #make it set to return
    word_set = set(scripts.dm_to_df(response)['word'])
    return word_set

def jaccardSim(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return intersection/union

def getPearsons(df_col1, df_col2):
    #arguments are pandas columns of calculated similarities
    arr1 = df_col1.to_numpy()
    arr2 = df_col2.to_numpy()
    return pearsonr(arr1, arr2)

def runFileApiCall(file, call, n):
    df = pd.read_csv(file)
    df["V4"] = 0
    for i,row in df.iterrows():
        w1 = row["V1"]
        w2 = row["V2"]
        w1_set = getSimilar(w1, call, n)
        w2_set = getSimilar(w2, call, n)
        row["V4"] = jaccardSim(w1_set, w2_set)
    return getPearsons(df["V3"], df["V4"])
    
