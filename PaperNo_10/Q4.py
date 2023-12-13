import spacy
import neuralcoref

# Add it to the pipeline
nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)

# Printing the coreferences
doc1 = nlp('My sister has a dog. She loves him.')
print(doc1._.coref_clusters)