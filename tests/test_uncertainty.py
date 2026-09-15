import numpy as np
import pandas as pd
import pytest

from forgepulse.features import FEATURES, TARGETS, build_features
from forgepulse.model import ManufacturingModels


def test_uncertainty_uses_training_transform():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.uniform(100, 500, (30, len(FEATURES))), columns=FEATURES)
    for i, target in enumerate(TARGETS):
        frame[target] = frame[FEATURES[i]] * 2
    model = ManufacturingModels().fit(frame)
    _, uncertainty, _, _ = model.predict_with_uncertainty(frame.iloc[:3])
    scaled = model.performance.named_steps["scale"].transform(build_features(frame.iloc[:3]))
    trees = model.performance.named_steps["model"].estimators_[0].estimators_
    expected = np.std([t.predict(scaled) for t in trees], axis=0, ddof=1)
    np.testing.assert_allclose(uncertainty.iloc[:, 0], expected)
    frame.loc[0, FEATURES[0]] = np.nan
    with pytest.raises(ValueError):
        model.predict(frame)
