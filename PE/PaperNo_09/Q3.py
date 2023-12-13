from nltk.tokenize import word_tokenize

text = "Joe waited for the train. The train was late. Mary and Samantha took the bus. I looked for Mary and Samantha at the bus station."

print("Original string:")
print(text)
print("List of words:")
print(word_tokenize(text))
