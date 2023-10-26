from datamuse import datamuse
import pandas as pd
from scipy.stats import pearsonr
import os

api = datamuse.Datamuse()

def dm_to_df(datamuse_response):
    """Converts the json response of the datamuse API into a DataFrame
    :datamuse_response
        [{'word': 'foo', 'score': 100}, {'word': 'bar', 'score': 120}]
    """
    reformatted = {
        'word': [response['word'] for response in datamuse_response]
    }
    return pd.DataFrame.from_dict(reformatted)

def maxAPIcallset(word):
    response = api.words(ml=word,max=1000)
    return set(dm_to_df(response)['word'])


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
    if arg == 'rel_spc':
        response = api.words(rel_spc=word,max=count)
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
    df["sim"] = float(0)
    for i,row in df.iterrows():
        row = row.copy()
        w1 = row["W1"]
        w2 = row["W2"]
        df.loc[i,"sim"] = dmPairSim(w1,w2,call,n)
    return getPearsons(df["human_sim"], df["sim"])


def testBest():
    call = 'ml'
    max = 570
    data_sets = ['rg','wordsim']
    path = "datasets"
    wd = os.getcwd()
    path = os.path.join(wd, path)
    all_files = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith(".csv") and any(ele in file for ele in data_sets):
                relative_path = os.path.relpath(os.path.join(file, root))
                relative_path = os.path.join(relative_path, file)
                all_files.append(relative_path)
    for file in all_files:
        result = runFileApiCall(file, call, max)
        print('Result for', file, ':',result)

def getSimEvalCorr():
    data_sets = ['rg','mc','wordsim']
    path = "measures"
    wd = os.getcwd()
    path = os.path.join(wd, path)
    all_files = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith(".csv") and any(ele in file for ele in data_sets):
                relative_path = os.path.relpath(os.path.join(file, root))
                relative_path = os.path.join(relative_path, file)
                all_files.append(relative_path)
    for file in all_files:
        df = pd.read_csv(file,sep=';', names=['W1','W2','human_sim','sim'])
        print(file, ': ',getPearsons(df["human_sim"], df["sim"]))


if __name__ == "__main__":
    testBest()