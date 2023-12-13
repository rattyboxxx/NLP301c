
import spacy

nlp = spacy.load('en_core_web_lg')

text1 = "John lives in Canada"
text2 = "James lives in America, though he's not from there"

doc1 = nlp(text1)
doc2 = nlp(text2)

print(doc1.similarity(doc2))
