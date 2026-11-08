"""Unit tests for calibrated classifier head."""

import numpy as np
import pytest
from shutup_ads.model.classifier import CalibratedClassifier


def test_classifier_fit_and_predict(tmp_path):
    model_file = tmp_path / "test_clf.joblib"
    clf = CalibratedClassifier(model_path=model_file)
    assert not clf.is_fitted

    # 10 sports samples (class 0) centered at -1.0
    # 10 ad samples (class 1) centered at +1.0
    np.random.seed(42)
    X_sports = np.random.normal(loc=-1.0, scale=0.2, size=(15, 1024)).astype(np.float32)
    X_ads = np.random.normal(loc=+1.0, scale=0.2, size=(15, 1024)).astype(np.float32)

    X = np.vstack([X_sports, X_ads])
    y = np.array([0] * 15 + [1] * 15)

    clf.fit(X, y, save=True)
    assert clf.is_fitted
    assert model_file.exists()

    # Predict sports sample
    p_sports = clf.predict_ad_probability(X_sports[:2])
    assert p_sports is not None
    assert p_sports < 0.20

    # Predict ad sample
    p_ads = clf.predict_ad_probability(X_ads[:2])
    assert p_ads is not None
    assert p_ads > 0.80

    # Reload from disk
    clf2 = CalibratedClassifier(model_path=model_file)
    assert clf2.is_fitted
    assert clf2.predict_ad_probability(X_ads[:2]) > 0.80
