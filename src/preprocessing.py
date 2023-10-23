import re
import contractions
import nltk
from nltk.corpus import wordnet
from antonyms import word_antonym_replacer
from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()
rep_antonym = word_antonym_replacer()



replace_dict = {'0':'zero', '1':'one', '2':'two', '3':'three', '4':'four', '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine'}
replace_tens = {'0':'', '10':'ten','11':'elven','12':'twelve','13':'thirteen','14':'fourteen','15':'fifteen','16':'sixteen','17':'seventeen','18':'eightteen','19':'nineteen', 
'2':'twenty', '3':'thirty', '4':'fourty', '5':'fifty', '6':'sixty', '7':'seventy', '8':'eighty', '9':'ninety'}
replace_nozero = {'0':'','1':'one', '2':'two', '3':'three', '4':'four', '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine'}

def digittotext(sent):
    #replaces 3 digit numbers with text, cuts leading zeros, and changes decimals to format "point number number..."
    #numbers have to be separate to be changed, for example 'm16' will not be changed

    x = sent
    match = re.findall(r'\.\d+\b', x)
    #decimal
    if match: 
        for number in match:
            replacement = ' point'
            i = 1
            while i < len(number):
                replacement += ' ' + replace_dict[number[i]]
                i += 1
            x = x.replace(number,replacement)

    #3digit
    match = re.findall(r'\b\d\d\d\b', x)
    if match:
        for number in match:
            replacement = ''
            if not number[0] == '0':
                replacement += replace_dict[number[0]] + ' hundred'
            if not (number[1] == '0' and number[2] == '0'):
                if number[1] == '1':
                    replacement += ' ' + replace_tens[number[1:3]]
                else:
                    replacement += ' ' + replace_tens[number[1]] + ' ' + replace_nozero[number[2]]
            x = x.replace(number,replacement)
    #2digit
    match = re.findall(r'\b\d\d\b', x)
    if match:
        for number in match:
            replacement = ''
            if number[0] == '1':
                replacement += replace_tens[number[0:2]]
            elif number[0] == '0':
                replacement += replace_dict[number[1]]
            else:
                replacement += replace_tens[number[0]] + ' ' + replace_nozero[number[1]]
        x = x.replace(number,replacement)
    #1digit
    match = re.findall(r'\b\d\b', x)
    if match:
        for number in match:
            replacement = replace_dict[number[0]]
        x = x.replace(number,replacement)

    x = x.replace('  ',' ') #double spaces

    return x

def removeStopwords(sentence):
    stopwords = list(set(nltk.corpus.stopwords.words('english')))
    stopwords.extend(['could', 'would']) #for some reason not included
    stopwords.remove('not') #leave not for antonyms
    stopwords.remove('no')
    x = sentence.split()
    filtered_sentence = [w for w in x if w.isalpha() and w not in stopwords]
    return filtered_sentence


def preprocess(sentence):
    #gets rid of relevant special chars 
    sentence = str(sentence).lower().replace('no one', 'none')
    sentence = digittotext(sentence)
    sentence = re.sub("[^A-Za-z0-9' ]+", '', sentence)
    sentence = contractions.fix(sentence)
    sentence = sentence.replace('cannot', 'can not')
    with_sw = sentence
    sentence = removeStopwords(sentence)
    sentence = rep_antonym.replace_negations(sentence)
    if 'not' in sentence:
        sentence.remove('not')
    elif 'no' in sentence:
        sentence.remove('no')
    sentence = [lemmatizer.lemmatize(w) for w in sentence]
    return sentence

def preprocess2(sentence):
    #very minimal preprocessing for fasttext, GloVe and word2vercor
    
    sentence = str(sentence).lower()
    sentence = contractions.fix(sentence)
    sentence = re.sub("[^A-Za-z0-9]+", '', sentence)
    return sentence

if __name__ == "__main__":
    print(preprocess('The ghost of Queen Victoria appears to me every night, I don’t know why, I don’t even like the royals.'))