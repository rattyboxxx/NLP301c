import nltk

from nltk.tokenize import WordPunctTokenizer
text = "Reset your password if you just can't remember your old one."
print("\nOriginal string:")
print(text)
result = WordPunctTokenizer().tokenize(text)
print("\nSplit all punctuation into separate tokens:")
print(result)