import numpy as np
import pandas as pd
from scipy import stats
import gensim
import sklearn.model_selection as cv
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.linear_model import Ridge
from statsmodels import robust
from sklearn.metrics import r2_score, make_scorer
from sklearn.decomposition import PCA

from word2vec_utils import GoogleVec
from reproduce_IAT import evaluate_embedding

sns.set_style('ticks')

g = GoogleVec()
word_vecs = g.wv

# vec_path = '/media/Freeman/data/word_embeddings/glove/glove.840B.300d.w2v'
# vec_path = '/media/Freeman/data/word_embeddings/glove/glove.twitter.27B.200d.w2v'
# vec_path = '/media/Freeman/data/word_embeddings/glove/glove.6B.300d.w2v'
# word_vecs = gensim.models.KeyedVectors.load_word2vec_format(vec_path, binary=False, limit=100000)

# vec_path = '../vectors/w2v_nytimes'
# word_vecs_nyt = gensim.models.Word2Vec.load(vec_path).wv

# vec_path = '../vectors/w2v_breitbart'
# word_vecs_breit= gensim.models.Word2Vec.load(vec_path).wv


def normalize(target):
    return '_'.join([t.capitalize() for t in target.split(' ')])

def scale_vec(x):
    return (x - np.median(x)) / robust.mad(x)

d0 = pd.read_csv('./data/wc_ratings_unnormalized_typed.csv')
d1 = pd.read_csv('./data/wc_validation2.csv')
d2 = pd.read_csv('./data/wc_validation.csv')
data = pd.concat([d0, d1, d2])
# data = data.ix[data['type'] != 'state']
# data = data.ix[(data['type'] != 'state') & (data['type'] != 'name')]
# data = data.ix[data['type'] == 'name']
# data = data.ix[data['type'] == 'occupation']

# data = d0

data = data.ix[[t in word_vecs for t in data.target]]

# not_state = np.array(data['type'] != 'state')

x = np.array(data.target)
X_vecs = word_vecs[x]

y_competence = np.array(data['competence'])
y_warmth = np.array(data['warmth'])

params = {'alpha': 10.0}
model = Ridge(**params)

model_c = copy.copy(model).fit(X_vecs, y_competence)
model_w = copy.copy(model).fit(X_vecs, y_warmth)


def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)

def drop_mat(X, v):
    proj = (X.dot(v) / np.dot(v,v) * v[:, np.newaxis]).T
    return X - proj

def sim(u, v):
    return np.dot(u, v) / np.sqrt(np.dot(u,u)*np.dot(v,v))
    
warmth_direction = model_w.coef_
competence_direction = model_c.coef_

def create_embedding(syn0, vocab, index2word):
    kv = gensim.models.KeyedVectors(300)
    kv.syn0 = np.array(syn0)
    kv.vocab = vocab
    kv.index2word = index2word
    kv.init_sims(replace=True)
    return kv

def remove_bias(wv, directions):
    vecs = word_vecs.syn0
    remove = np.array(directions)
    if len(remove.shape) == 1:
        remove = remove[np.newaxis, :]
    for i in range(remove.shape[0]):
        vecs = drop_mat(vecs, remove[i])
        remove = drop_mat(remove, remove[i])
    return create_embedding(vecs, wv.vocab, wv.index2word)

word_vecs_debiased = remove_bias(word_vecs, [warmth_direction, competence_direction])
word_vecs_debiased_warmth = remove_bias(word_vecs, warmth_direction)
word_vecs_debiased_competence = remove_bias(word_vecs, competence_direction)

# pca = PCA().fit(word_vecs[european_names + african_american_names])
pca = PCA().fit(X_vecs)
comps = [c for c in pca.components_[:20]
         if np.abs(sim(c, warmth_direction)) > 0.15
         or np.abs(sim(c, competence_direction)) > 0.15]
print(len(c))
word_vecs_debiased_new = remove_bias(word_vecs, comps)


print('word vecs normal')
evaluate_embedding(word_vecs)

print('')
print('word vecs debiased warmth + competence')
evaluate_embedding(word_vecs_debiased)

print('')
print('word vecs debiased warmth')
evaluate_embedding(word_vecs_debiased_warmth)

print('')
print('word vecs debiased competence')
evaluate_embedding(word_vecs_debiased_competence)

print('')
print('word vecs debiased PCA')
evaluate_embedding(word_vecs_debiased_new)
