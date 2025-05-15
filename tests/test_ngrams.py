#Autor: Ernesto Juárez Torres A01754887

def test_ngrams_bigram(tiny_corpus, ngram_features):
    X = ngram_features(tiny_corpus, n=(2,2), max_features=20)
    # puede haber 0‑features si no hay bigramas repetidos
    assert X.shape[0] == 2
