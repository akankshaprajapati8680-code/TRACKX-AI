"""
dashboard.py - basic Streamlit dashboard for TrackX AI.

Run:  streamlit run src/dashboard.py

Sources:
  * Simulated   : generates a SIMULATED waveform on demand (demo mode)
  * Upload CSV  : CSV of one waveform (column 'value' or first column)
  * Live (ESP32): placeholder - needs hardware + serial/Wi-Fi bridge (TODO)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from src.classify import DEFECT, NORMAL, TrackXClassifier, train_from_simulated
from src.features import extract_features
from src.simulate import FS, make_waveform

MODEL_PATH = Path("models/trackx_rf_SIMULATED.joblib")

st.set_page_config(page_title="TrackX AI", page_icon="🚆", layout="wide")
st.title("🚆 TrackX AI - Rail Defect Detection (prototype)")
st.warning("Prototype. Demo data is SIMULATED. Not for railway safety decisions - "
           "validate against established inspection methods first.")


@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return TrackXClassifier.load(MODEL_PATH)
    from src.simulate import generate_dataset  # first run: build data + model
    import numpy as _np
    Path("data/simulated").mkdir(parents=True, exist_ok=True)
    if not Path("data/simulated/waveforms_SIMULATED.npz").exists():
        X, meta = generate_dataset()
        _np.savez_compressed("data/simulated/waveforms_SIMULATED.npz", waveforms=X, fs=FS)
        meta.to_csv("data/simulated/metadata_SIMULATED.csv", index=False)
    clf, _ = train_from_simulated()
    return clf


clf = get_model()
if "log" not in st.session_state:
    st.session_state.log = []

source = st.sidebar.radio("Waveform source", ["Simulated", "Upload CSV", "Live (ESP32)"])
wave, truth = None, None

if source == "Simulated":
    kind = st.sidebar.selectbox("Simulated case", ["normal", "defect", "bad_coupling"])
    if st.sidebar.button("Generate & classify") or "last_wave" not in st.session_state:
        w, _ = make_waveform(kind, np.random.default_rng())
        st.session_state.last_wave = (w, kind)
        st.session_state.new = True
    wave, truth = st.session_state.last_wave
elif source == "Upload CSV":
    f = st.sidebar.file_uploader("Waveform CSV", type="csv")
    if f is not None:
        df = pd.read_csv(f)
        wave = df["value"].values if "value" in df else df.iloc[:, 0].values
        st.session_state.new = True
else:
    st.info("Live mode needs the ESP32 hardware. See firmware/ and README.")

if wave is not None:
    feats = extract_features(wave, FS)
    res = clf.predict_one(feats)

    c1, c2 = st.columns([2, 1])
    with c1:
        t_us = np.arange(len(wave)) / FS * 1e6
        st.line_chart(pd.DataFrame({"amplitude": wave}, index=pd.Index(t_us, name="time (us)")))
    with c2:
        label = res["label"]
        if label == NORMAL:
            st.success(f"✅ {label} - logged, no action needed")
        elif label == DEFECT:
            st.error(f"🚨 {label} - ALERT: further inspection needed")
        else:
            st.warning(f"❓ {label} - re-measure or inspect ({res['reason']})")
        st.metric("P(defect)", f"{res['p_defect']:.2f}")
        st.metric("Confidence", f"{res['confidence']:.2f}")
        st.metric("SNR (dB)", f"{feats['snr_db']:.1f}")
        if truth:
            st.caption(f"Simulated ground truth: {truth}")

    if st.session_state.pop("new", False):   # log once per new reading, not per rerun
        st.session_state.log.append({"result": res["label"],
                                     "p_defect": round(res["p_defect"], 3),
                                     "snr_db": round(feats["snr_db"], 1)})
    with st.expander("Extracted features"):
        st.dataframe(pd.DataFrame([feats]))

st.subheader("Result log")
if st.session_state.log:
    st.dataframe(pd.DataFrame(st.session_state.log).tail(50), use_container_width=True)
