"""
features.py - hand-crafted features from one ultrasonic waveform.

Features (matching the TrackX AI plan): echo amplitude, arrival time, energy,
spectrum (dominant frequency, centroid, bandwidth), SNR, attenuation, echo count.
Raw waveforms are kept separately so a 1D-CNN can be tried later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert

FS_DEFAULT = 10e6
BANG_END_US = 12.0      # ignore the initial transmit pulse before this time
NOISE_TAIL_FRAC = 0.15  # last 15 % of the capture is assumed echo-free (noise floor)

FEATURE_NAMES = [
    "peak_amp", "peak_time_us", "energy", "end_echo_amp", "attenuation_ratio",
    "n_echoes", "first_echo_amp", "dom_freq_mhz", "centroid_mhz", "bandwidth_mhz",
    "snr_db", "noise_rms",
]


def extract_features(wave, fs=FS_DEFAULT):
    wave = np.asarray(wave, dtype=float)
    n = len(wave)
    t_us = np.arange(n) / fs * 1e6
    wave = wave - wave.mean()

    env = np.abs(hilbert(wave))
    bang_mask = t_us < BANG_END_US
    bang_amp = env[bang_mask].max() if bang_mask.any() else 1.0

    tail = wave[int(n * (1 - NOISE_TAIL_FRAC)):]
    noise_rms = float(np.sqrt(np.mean(tail ** 2))) + 1e-12

    echo_env = env.copy()
    echo_env[bang_mask] = 0.0
    thr = 6 * noise_rms
    peaks, props = find_peaks(echo_env, height=thr, distance=int(0.8e-6 * fs))

    if len(peaks):
        heights = props["peak_heights"]
        main = peaks[np.argmax(heights)]
        peak_amp = float(heights.max())
        peak_time = float(t_us[main])
        first_amp = float(heights[0])
        end_echo_amp = float(heights[-1])          # latest echo ~ back wall
    else:
        peak_amp = peak_time = first_amp = end_echo_amp = 0.0

    energy = float(np.sum(wave[~bang_mask] ** 2) / fs * 1e6)
    attenuation = end_echo_amp / (bang_amp + 1e-12)

    seg = wave[~bang_mask]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1 / fs) / 1e6
    p = spec ** 2
    centroid = float((freqs * p).sum() / (p.sum() + 1e-12))
    bw = float(np.sqrt(((freqs - centroid) ** 2 * p).sum() / (p.sum() + 1e-12)))
    dom = float(freqs[np.argmax(spec)])

    snr_db = float(20 * np.log10((peak_amp + 1e-12) / noise_rms))

    return {
        "peak_amp": peak_amp, "peak_time_us": peak_time, "energy": energy,
        "end_echo_amp": end_echo_amp, "attenuation_ratio": float(attenuation),
        "n_echoes": int(len(peaks)), "first_echo_amp": first_amp,
        "dom_freq_mhz": dom, "centroid_mhz": centroid, "bandwidth_mhz": bw,
        "snr_db": snr_db, "noise_rms": noise_rms,
    }


def feature_matrix(waves, fs=FS_DEFAULT):
    return pd.DataFrame([extract_features(w, fs) for w in waves])[FEATURE_NAMES]
