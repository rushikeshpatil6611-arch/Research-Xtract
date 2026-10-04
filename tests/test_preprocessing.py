from researchxtract.core.preprocessing import analyze, stem, tokenize


def test_tokenize_lowercases_and_splits_hyphens():
    assert tokenize("GPS-denied 3D Vision") == ["gps", "denied", "3d", "vision"]


def test_stopwords_and_short_tokens_removed():
    assert analyze("the CNN of a x in it") == ["cnn"]


def test_stemming_makes_plural_and_ing_forms_match():
    assert stem("images") == stem("image")
    assert stem("learning") == "learn"
    assert stem("anomalies") == "anomaly"
    assert stem("analysis") == "analysis"


def test_same_pipeline_for_query_and_document():
    assert analyze("Medical Images") == analyze("medical image")
