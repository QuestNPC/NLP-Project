from nltk.corpus import wordnet



class word_antonym_replacer(object):
    def replace(self, word, pos=None):
        #antonyms = set()
        ants = {}
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas():
                for antonym in lemma.antonyms():
                    #antonyms.add(antonym.name())

                    #get antonyms and how frequent they are
                    try:
                        ants[antonym.name()] += antonym.count()
                    except KeyError:
                        ants[antonym.name()] = antonym.count()
        '''
        if len(antonyms) > 0:
            return antonyms.pop()
        '''
        #most common antonym
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
    
if __name__ == "__main__":
    rep_antonym = word_antonym_replacer()
    print(rep_antonym.replace('lose'))