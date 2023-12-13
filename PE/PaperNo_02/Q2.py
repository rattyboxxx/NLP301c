import nltk

text = ''''
Joe waited for the train. The train was late. 
Mary and Samantha took the bus. 
I looked for Mary and Samanthasize at the bus size.
'''

word = "Mary"


def percent(word, text):
    text = nltk.tokenize.word_tokenize(text.lower())
    text = [word for word in text if word.isalnum()]
    return text.count(word.lower()) / len(text)


print(percent(word, text))
