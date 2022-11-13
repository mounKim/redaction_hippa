from gensim.models import Word2Vec


def tokenize_alldata(dataset, tokenizer):
    tokens = []
    for data in dataset:
        tokens.append(tokenizer.tokenize(data))
    return tokens


def make_word2vec(sentences, min_count):
    model = Word2Vec(sentences=sentences, vector_size=200, min_count=min_count)
    return model
