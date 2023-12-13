import en_core_web_sm
import textacy
nlp = en_core_web_sm.load()

text = ("I may bake a cake for my birthday. The talk will introduce reader about Use of baking")
# Regex pattern to identify verb phrase
pattern = r'(<VERB>?<ADV>*<VERB>+)'
doc = textacy.make_spacy_doc(text, lang='en_core_web_sm')

# Finding matches
for ele in textacy.extract.matches.regex_matches(doc, pattern):
    print(ele.text)
