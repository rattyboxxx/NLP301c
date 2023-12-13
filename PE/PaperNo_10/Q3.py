from nltk.tokenize import word_tokenize, sent_tokenize

text = "Joe waited for the train. The train was late. Mary and Samantha took the bus. I looked for Mary and Samantha at the bus station."

print("Original string:")
print(text)
print("Tokenize words sentence wise:")
for ele in sent_tokenize(text):
    print(word_tokenize(ele))
