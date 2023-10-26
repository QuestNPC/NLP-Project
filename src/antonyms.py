from nltk.corpus import wordnet

#adapted from tutorialspoint template: https://www.tutorialspoint.com/natural_language_toolkit/natural_language_toolkit_synonym_antonym_replacement.htm

class word_antonym_replacer(object):
    def replace(self, word):
        #antonyms = set()
        ants = {}
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    #antonyms.add(antonym.name())
                    #get antonyms and how frequent they are
                    try:
                        ants[antonym.name()] += antonym.count()
                    except KeyError:
                        ants[antonym.name()] = antonym.count()
        
        #most common antonym as lemma
        if len(ants) > 0:
            return max(zip(ants.values(), ants.keys()))[1]
        else:
            return None
      
    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i+1 < l:
                ant = self.replace(sent[i+1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words
