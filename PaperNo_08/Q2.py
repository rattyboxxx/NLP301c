import nltk
from nltk.corpus import brown

res = {}

for cate in brown.categories():
    news_text = brown.words(categories=cate)
    fdist = nltk.FreqDist(w.lower() for w in news_text)

    for key, value in fdist.items():
        res[key] = res.get(key, 0) + fdist[key]

print(sorted([word for word in res.keys() if res[word] >= 3]))
