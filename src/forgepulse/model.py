from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, IsolationForest
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import TARGETS, build_features


class ManufacturingModels:
    """Supervised performance models plus an unsupervised anomaly detector."""

    def __init__(self) -> None:
        base = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.performance = Pipeline([("scale", StandardScaler()), ("model", MultiOutputRegressor(base))])
        self.anomaly = IsolationForest(
            n_estimators=200,
            contamination=0.05,
            random_state=42,
        )
        self.version = "3.0.0"

    def fit(self, frame: pd.DataFrame) -> "ManufacturingModels":
        x = build_features(frame)
        y = frame[TARGETS]
        self.performance.fit(x, y)
        self.anomaly.fit(x)
        return self

    def predict(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        prediction, uncertainty, score, flag = self.predict_with_uncertainty(frame)
        return prediction, score, flag

    def predict_with_uncertainty(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        x = build_features(frame)
        prediction = pd.DataFrame(self.performance.predict(x), columns=TARGETS, index=x.index)
        estimators = self.performance.named_steps["model"].estimators_
        tree_predictions = []
        for estimator in estimators:
            tree_predictions.append(
                pd.DataFrame(
                    [tree.predict(x) for tree in estimator.estimators_],
                    columns=[estimator.estimators_[0].n_features_in_],
                )
            )
        uncertainty = pd.DataFrame(index=x.index)
        for target, estimator in zip(TARGETS, estimators):
            values = [tree.predict(x) for tree in estimator.estimators_]
            uncertainty[f"{target}_std"] = pd.DataFrame(values).std(axis=0).to_numpy()
        score = pd.Series(self.anomaly.decision_function(x), index=x.index, name="anomaly_score")
        flag = pd.Series(self.anomaly.predict(x) == -1, index=x.index, name="anomaly")
        return prediction, uncertainty, score, flag

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "ManufacturingModels":
        return joblib.load(path)
