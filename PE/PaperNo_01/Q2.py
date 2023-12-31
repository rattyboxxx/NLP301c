import nltk

text = '''               
William Shakespeare's name is synonymous with many of the famous lines he wrote in his plays and prose. Yet his poems are not nearly as recognizable to many as the characters and famous monologues from his many plays.

In Shakespeare's era (1564-1616), it was not profitable but very fashionable to write poetry. It also provided credibility to his talent as a writer and helped to enhance his social standing. It seems writing poetry was something he greatly enjoyed and did mainly for himself at times when he was not consumed with writing a play. Because of their more private nature, few poems, particularly long-form poems, have been published.

The two longest works that scholars agree were written by Shakespeare are entitled Venus and Adonis and The Rape of Lucrece. Both dedicated to the Honorable Henry Wriothesley, Earl of Southampton, who seems to have acted as a sponsor and encouraging benefactor of Shakespeare's work for a brief time.

Both of these poems contain dozens of stanzas and comment on the depravity of unwanted sexual advances, showing themes throughout of guilt, lust, and moral confusion. In Venus and Adonis, an innocent Adonis must reject the sexual advances of Venus. Conversely in The Rape of Lucrece, the honorable and virtuous wife Lucrece is raped a character overcome with lust, Tarquin. The dedication to Wriothesley is much warmer in the second poem, suggesting a deepening of their relationship and Shakespeare's appreciation of his support.
'''


def freq_non_stopwords(text, number):
    stopwords = nltk.corpus.stopwords.words('english')
    clean_list = [w for w in nltk.tokenize.word_tokenize(
        text) if w.lower() not in stopwords]
    freqdist = nltk.probability.FreqDist(clean_list)
    return freqdist.most_common(number)


print(freq_non_stopwords(text, 50))
