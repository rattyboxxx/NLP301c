def set_txt(text, vocabulary):
    return set([word for word in set(text) if word not in set(vocabulary)])