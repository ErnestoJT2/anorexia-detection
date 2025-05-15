#Autor: Ernesto Juárez Torres A01754887

def test_sentiment_range(tiny_corpus, sentiment_vector):
    v = sentiment_vector(tiny_corpus)
    assert v.shape == (2,1)
    assert ((v >= 0) & (v <= 1)).all()
