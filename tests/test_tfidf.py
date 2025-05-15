#Autor: Ernesto Juárez Torres A01754887

def test_tfidf_basic(tiny_corpus, tfidf_features):
    X = tfidf_features(tiny_corpus, max_features=10)
    assert X.shape == (2, 8) or X.shape[0] == 2
    # todas las celdas son no negativas
    assert (X.data >= 0).all()
