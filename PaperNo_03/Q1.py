from nltk.tokenize import word_tokenize

def percent(word, text):
    text = [word for word in word_tokenize(text.lower()) if word.isalpha()]
    return text.count(word) / len(text)