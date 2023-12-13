# Need to install spacy library
# pip install spacy

import spacy

text = "My sister has a dog and she loves him" 

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)

pronouns = []
objects = []
tmp = []
object_list = ["i", "we", "you", "they", "he", "she", "it", "me", "us", "them", "him", "her"]
for token in doc:
    if token.pos_ in ["PRON", "DET", "NOUN"] and token.text.lower() not in object_list :
        tmp.append(token.text)
    elif token.text.lower() in object_list:
        objects.append(token.text)
    else:
        if len(tmp) > 0:
            pronouns.append(" ".join(tmp))
            tmp = []

for i in range(len(pronouns)):
    print([pronouns[i], objects[i]])
