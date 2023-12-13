from nltk.tokenize import word_tokenize, sent_tokenize

text = "Joe waited for the train. The train was late. Mary and Samantha took the bus. I looked for Mary and Samantha at the bus station."

print("Original string:")
print(text)
print("Sentence-tokenized copy in a list:")
print(sent_tokenize(text))
print("Read the list:")
for ele in sent_tokenize(text):
    print(ele)
