from src.model_training import train_models

def test_train_models_output():
    result = train_models("data/ASD_meta_abundance.csv")
    assert isinstance(result, tuple)
    assert len(result) == 6

    model, X_test, y_test, y_pred, cv_score, feature_names = result
    assert hasattr(model, "predict")
    assert X_test.shape[0] == len(y_test) == len(y_pred)
    assert isinstance(cv_score, float)
