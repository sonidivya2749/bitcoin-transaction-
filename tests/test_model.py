import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "models") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "models"))
if str(PROJECT_ROOT / "dashboard") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))


FEATURE_COLS = ["tx_count", "avg_amount", "total_amount", "avg_fee", "avg_time_gap_seconds"]


def make_synthetic_wallet_features(n_normal=200, n_suspicious=10, seed=42):
    
    rng = np.random.default_rng(seed)

    normal = pd.DataFrame({
        "wallet": [f"wallet_{i:04d}" for i in range(n_normal)],
        "tx_count": rng.integers(1, 30, n_normal),
        "avg_amount": rng.uniform(0.05, 2.0, n_normal),
        "avg_fee": rng.uniform(0.0001, 0.001, n_normal),
        "avg_time_gap_seconds": rng.uniform(300, 86400, n_normal),
    })
    normal["total_amount"] = normal["tx_count"] * normal["avg_amount"]

    suspicious = pd.DataFrame({
        "wallet": [f"wallet_S{i:03d}" for i in range(n_suspicious)],
        "tx_count": rng.integers(300, 800, n_suspicious),
        "avg_amount": rng.uniform(0.0001, 0.002, n_suspicious),
        "avg_fee": rng.uniform(0.00001, 0.00005, n_suspicious),
        "avg_time_gap_seconds": rng.uniform(1, 30, n_suspicious),
    })
    suspicious["total_amount"] = suspicious["tx_count"] * suspicious["avg_amount"]

    df = pd.concat([normal, suspicious], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture
def synthetic_features():
    return make_synthetic_wallet_features()


def train_isolation_forest(df, contamination=0.05, random_state=42):
    X = df[FEATURE_COLS].values
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state)
    model.fit(X)
    return model, X


class TestFeatureExtraction:
    def test_synthetic_features_have_required_columns(self, synthetic_features):
        for col in FEATURE_COLS:
            assert col in synthetic_features.columns, f"Missing required feature column: {col}"

    def test_no_negative_transaction_counts(self, synthetic_features):
        assert (synthetic_features["tx_count"] > 0).all()

    def test_no_null_values_in_features(self, synthetic_features):
        assert synthetic_features[FEATURE_COLS].isnull().sum().sum() == 0


class TestIsolationForestModel:
    def test_model_trains_without_error(self, synthetic_features):
        model, X = train_isolation_forest(synthetic_features)
        assert model is not None

    def test_model_flags_some_wallets_as_anomalous(self, synthetic_features):
        model, X = train_isolation_forest(synthetic_features)
        predictions = model.predict(X)
        n_flagged = (predictions == -1).sum()
        assert n_flagged > 0, "Model flagged zero wallets -- contamination setting or features may be broken."

    def test_flag_rate_roughly_matches_contamination(self, synthetic_features):
        contamination = 0.05
        model, X = train_isolation_forest(synthetic_features, contamination=contamination)
        predictions = model.predict(X)
        flag_rate = (predictions == -1).mean()
        # Isolation Forest's contamination setting is a target, not exact --
        # allow reasonable tolerance either side.
        assert abs(flag_rate - contamination) < 0.05

    def test_planted_suspicious_wallets_are_caught(self, synthetic_features):
        
        model, X = train_isolation_forest(synthetic_features)
        predictions = model.predict(X)
        synthetic_features = synthetic_features.copy()
        synthetic_features["predicted_anomaly"] = predictions == -1

        planted = synthetic_features[synthetic_features["wallet"].str.startswith("wallet_S")]
        catch_rate = planted["predicted_anomaly"].mean()
        assert catch_rate >= 0.6, f"Only caught {catch_rate:.0%} of the obviously suspicious planted wallets."

    def test_suspicion_score_is_in_valid_range(self, synthetic_features):
        model, X = train_isolation_forest(synthetic_features)
        raw = -model.decision_function(X)
        lo, hi = raw.min(), raw.max()
        suspicion = (raw - lo) / (hi - lo) * 100 if hi > lo else np.zeros_like(raw)
        assert suspicion.min() >= 0
        assert suspicion.max() <= 100

    def test_deterministic_with_fixed_seed(self, synthetic_features):
    
        model1, X = train_isolation_forest(synthetic_features, random_state=42)
        model2, _ = train_isolation_forest(synthetic_features, random_state=42)
        preds1 = model1.predict(X)
        preds2 = model2.predict(X)
        assert (preds1 == preds2).all()


class TestModelFileIfPresent:
    

    @pytest.fixture
    def real_model_path(self):
        path = PROJECT_ROOT / "models" / "isolation_forest_model.pkl"
        if not path.exists():
            pytest.skip("No trained model file found yet -- run models/model.py first.")
        return path

    def test_real_model_loads(self, real_model_path):
        import joblib
        model = joblib.load(real_model_path)
        assert hasattr(model, "predict")
        assert hasattr(model, "decision_function")

    def test_real_wallet_scores_file_is_well_formed(self):
        scores_path = PROJECT_ROOT / "models" / "wallet_scores.csv"
        if not scores_path.exists():
            pytest.skip("wallet_scores.csv not generated yet.")
        df = pd.read_csv(scores_path)
        assert "wallet" in df.columns
        assert "suspicion_score" in df.columns
        assert "is_anomaly" in df.columns
        assert df["is_anomaly"].isin(["Yes", "No"]).all()
        assert (df["suspicion_score"] >= 0).all()
        assert (df["suspicion_score"] <= 100).all()
