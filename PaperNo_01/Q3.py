from nltk.tokenize import WordPunctTokenizer

text = "Reset your password if you just can't remember your old one."

print("Original string:")
print(text)
print("Split all punctuation into separate tokens:")
print(WordPunctTokenizer().tokenize(text))
