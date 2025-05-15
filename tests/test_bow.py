#Autor: Ernesto Juárez Torres A01754887

def test_bow_shape(tiny_corpus, bow_features):
    X = bow_features(tiny_corpus, max_features=10)
    assert X.shape[0] == 2
    expected = sum(len(doc.split()) for doc in tiny_corpus)  # = 8
    assert X.sum() == expected
