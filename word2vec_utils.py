
import gensim
import os
import numpy as np
import enchant


google_vector_path = '../data/GoogleNews-vectors-negative300.bin'
twitter_vector_path = '../data/word2vec_twitter_model.bin'



class GoogleVec:
    
    def __init__(self):
       model =  gensim.models.KeyedVectors.load_word2vec_format(google_vector_path, binary = True)
       word_vecs = model.wv
       self.wv = word_vecs
       del model
       self.d  = enchant.Dict('en_US')

    
    def get_w2v(self, name, word=False, norm=False):

        if len(name.split(' ')) > 1:
            name = name.replace(' ', '_')
        try:
            w2v = self.wv[name]
            if(word):
                return name
            elif(norm):
                return w2v/np.linalg.norm(w2v)
            return w2v
        except:
            print("{} is not in the dictionary, trying splitting the word".format(name))
        corr_name = name.split('_')
        try:
            name = tuple(name)
            w2v = self.wv[tuple(name)]
            if(word):
                return name
            elif(norm):
                return w2v/np.linalg.norm(w2v)
            return w2v
        except:
            print("Not all component words in list")

    def similar(self, name1, name2):
        try:
            return self.wv.n_similarity(name1.split(' ').join('_'), name2.split(' ').join('_'))
        except: 
            return self.wv.n_similarity(name1.split(' '), name2.split(' '))

class TwitterVec:

    def __init__(self):

        model =  gensim.models.KeyedVectors.load_word2vec_format(twitter_vector_path, binary = True, unicode_errors = 'ignore')
        self.wv = model.wv
        del model
        self.d = enchant.Dict('en_US')

    def get_w2v(self, name):
        name_list = name.split(' ')
        try:
            w2v = self.wv[tuple(name_list)]
            return w2v
        except:
            print("{} is not in the dictionary".format(name))
    
    def similar(self, name1, name2):
        return self.wv.n_similarity(name1.split(), name2.split())


#Will Spellcheck the word... eventually
def spellcheck(word):
    return word
