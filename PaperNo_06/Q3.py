# Need to install matplotlib library
# pip install matplotlib

from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

text = "He would also attend the opening ceremony for the construction of the U.S. Embassy complex in Cau Giay District, as well as meeting students, teachers and scientists at the Hanot University of Science and Technology"
text = word_tokenize(text)
fdist = FreqDist(word for word in text if word.isalpha() and len(word) < 4)

print(list(fdist.keys()))

x, y = [], []
res = sorted(fdist.items(), key=lambda x: x[-1], reverse=True)
for ele in res:
    x.append(ele[0])
    y.append(ele[1])

plt.plot(y)
plt.xticks([i for i in range(len(x))], x, rotation=90)
plt.xlabel("Samples")
plt.ylabel("Counts")
plt.grid()
plt.show()

