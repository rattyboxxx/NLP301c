from nltk.tokenize import word_tokenize
import re

def percent(word, text):
    text = re.sub('[^a-zA-Z0-9\\s]+', '', text)
    text = [word for word in word_tokenize(text.lower())]
    return str(text.count(word) / len(text) * 100) + "%"

# Test the function
# text = "The quick, brown fox jumps over the lazy dog."
# word = "the"
# print(percent(word, text))