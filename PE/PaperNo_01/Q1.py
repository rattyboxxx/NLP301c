from nltk.tokenize import word_tokenize

text = ''''
Joe waited for the train. The train was late. 
Mary and Samantha took the bus. 
I looked for Mary and Samanthasize at the bus size.
'''

print([ele for ele in word_tokenize(text) if ele.endswith('ize')])