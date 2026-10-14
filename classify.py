"""
classify.py - Random Forest + "Inconclusive" rule.

Three outputs, as in the TrackX AI design:
    Normal | Potentially defective | Inconclusive

A reading is INCONCLUSIVE when
    (a) signal quality is poor : SNR below `min_snr_db`, or no echo found, or
    (b) the model is unsure    : highest class probability below `conf_threshold`.

NOTE: trained on SIMULATED data unless you supply real labelled data.
Metrics from simulated data are NOT real-world performance.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from .features import FEATURE_NAMES, feature_matrix, extract_features

NORMAL, DEFECT, INCONCLUSIVE = "Normal", "Potentially defective", "Inconclusive"


@dataclass
class Rule:
    conf_threshold: float = 0.80
    min_snr_db: float = 12.0


class TrackXClassifier:
    def __init__(self, rule=None, seed=42):
        self.rule = rule or Rule()
        self.model = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X[FEATURE_NAMES], y)
        return self

    def predict_one(self, feats):
        row = pd.DataFrame([feats])[FEATURE_NAMES]
        proba = self.model.predict_proba(row)[0]
        classes = list(self.model.classes_)
        p_def = float(proba[classes.index(1)])
        conf = float(proba.max())

        if feats["n_echoes"] == 0 or feats["snr_db"] < self.rule.min_snr_db:
            return {"label": INCONCLUSIVE, "reason": "poor signal quality (low SNR / no echo)",
                    "p_defect": p_def, "confidence": conf}
        if conf < self.rule.conf_threshold:
            return {"label": INCONCLUSIVE, "reason": "model not confident enough",
                    "p_defect": p_def, "confidence": conf}
        label = DEFECT if p_def >= 0.5 else NORMAL
        return {"label": label, "reason": "", "p_defect": p_def, "confidence": conf}

    def predict_waveform(self, wave, fs=10e6):
        return self.predict_one(extract_features(wave, fs))

    def predict_df(self, X):
        return [self.predict_one(r.to_dict())["label"] for _, r in X.iterrows()]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def evaluate(true_defect, pred_labels):
    """true_defect: 0/1 ground truth. pred_labels: strings from this module."""
    y = np.asarray(true_defect)
    p = np.asarray(pred_labels)
    inconc = p == INCONCLUSIVE
    decided = ~inconc
    pred_bin = (p == DEFECT).astype(int)
    out = {"n": int(len(y)), "inconclusive_share": float(inconc.mean())}
    if decided.sum():
        yd, pd_ = y[decided], pred_bin[decided]
        out["recall_defect"] = float(recall_score(yd, pd_, zero_division=0))
        out["precision_defect"] = float(precision_score(yd, pd_, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(yd, pd_, labels=[0, 1]).ravel()
        out["false_alarm_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
        out["missed_defects"] = int(fn)
    return out


def train_from_simulated(data_dir="data/simulated", out_model="models/trackx_rf_SIMULATED.joblib"):
    d = Path(data_dir)
    npz = np.load(d / "waveforms_SIMULATED.npz")
    meta = pd.read_csv(d / "metadata_SIMULATED.csv")
    X = feature_matrix(npz["waveforms"], float(npz["fs"]))

    # Train only on clean normal/defect; bad_coupling is kept for testing the rule.
    clean = (meta["kind"] != "bad_coupling").values
    Xc = X[clean].reset_index(drop=True)
    mc = meta[clean].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    tr_idx, te_idx = next(gss.split(Xc, mc["has_defect"], groups=mc["session"]))

    clf = TrackXClassifier().fit(Xc.iloc[tr_idx], mc["has_defect"].iloc[tr_idx])

    # held-out sessions (unseen) + all bad-coupling captures
    X_test = pd.concat([Xc.iloc[te_idx], X[~clean]], ignore_index=True)
    y_test = np.concatenate([mc["has_defect"].iloc[te_idx].values,
                             np.zeros((~clean).sum(), dtype=int)])
    res = evaluate(y_test, clf.predict_df(X_test))

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    clf.save(out_model)
    return clf, res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/simulated")
    ap.add_argument("--model", default="models/trackx_rf_SIMULATED.joblib")
    a = ap.parse_args()
    _, res = train_from_simulated(a.data, a.model)
    print("SIMULATED-DATA metrics (not real-world performance):")
    for k, v in res.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
