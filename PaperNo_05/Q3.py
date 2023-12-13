texts= [" Photography is an excellent hobby to pursue ",
        " Photographers usually develop patience, calmnesss",
        " You can try Photography with any good mobile too"]

import nltk
# We prepare a list containing lists of tokens of each text
all_tokens=[]
for text in texts:
  tokens=[]
  raw=nltk.wordpunct_tokenize(text.strip())
  for token in raw:
    tokens.append(token)
  all_tokens.append(tokens)

# Import and fit the model with data
from gensim.models import Word2Vec
model=Word2Vec(all_tokens, min_count=1)

# Visualizing the word embedding
from sklearn.decomposition import PCA
from matplotlib import pyplot

col = ['col' + str(i) for i in range(len(model.wv[0]))]
import pandas as pd
import numpy as np

X = pd.DataFrame([], columns=col)
for idx in range(len(model.wv)):
  X.loc[idx] = model.wv[idx]

# PCA down to 2D
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(set(sum(all_tokens, [])))
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()