from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk

text = nltk.corpus.nps_chat.words()

fdist = FreqDist(word for word in text if word.isalpha() and len(word) == 4)

print(sorted(fdist.items(), key=lambda x: x[-1], reverse=True))
