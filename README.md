# 🚆 TrackX AI - AI-Based Smart Railway Track Defect Detection

**AI Synergy Hackathon 2026 · Team ORIONS**
Akanksha (KIET Group of Institutions) · Piyush (Sharda University) · Akash (KIET Group of Institutions)

> *Detect defects early, inspect smarter, keep tracks safe.*

> ⚠️ **Honest status:** TrackX AI is a **prototype to assist inspection**. All data in this repository is **simulated**
> (files are labelled `*_SIMULATED*`). Metrics from simulated data are **not** real-world performance.
> Real performance needs real labelled data (NDT lab / public datasets) and validation against established
> inspection methods before any railway safety use. The firmware is an untested skeleton.

## The idea

Defects found late can grow into serious failures, manual inspection is slow and depends on human judgement,
and raw ultrasonic data takes expert time to analyse. (5,353 rail fractures were reported on Indian Railways in
2008-09 - Lok Sabha Unstarred Question No. 2, 19 Nov 2009; an older official figure.)

Two ESP32 nodes send and receive ultrasonic waves through the rail (Node A pulse-echo, Node B pitch-catch).
A pulser-receiver, signal conditioning and ADC capture the waveform; it is sent over Wi-Fi / ESP-NOW to a laptop,
where an AI model classifies it and a dashboard shows results and alerts.

| AI output | Action |
|---|---|
| **Normal** | Logged, no action needed |
| **Potentially defective** | Alert sent for further inspection |
| **Inconclusive** | Re-measure or inspect (weak/noisy signal or low model confidence) |

> Why not HC-SR04? It cannot reliably detect internal cracks in steel rails; a proper ultrasonic probe with
> pulser-receiver electronics is needed.

![architecture](docs/architecture.png)

## What is in this repo

```
trackx-ai/
├── README.md
├── docs/
│   ├── SRS.pdf                  Software Requirements Specification
│   ├── architecture.png         block diagram
│   └── simulated_waveforms.png  illustrative simulated waveforms
├── firmware/                    ESP32 Arduino skeleton (TX / RX nodes) - NOT hardware-tested
├── data/simulated/              simulated waveforms + metadata, clearly labelled
├── notebooks/
│   └── 01_features_and_classifier.ipynb
├── src/
│   ├── simulate.py              waveform generator (toy pulse-echo model)
│   ├── features.py              amplitude, energy, spectrum, SNR, attenuation ...
│   ├── classify.py              Random Forest + Inconclusive rule
│   └── dashboard.py             basic Streamlit dashboard
├── requirements.txt
└── LICENSE                      MIT
```

## Quick start

```bash
git clone https://github.com/<your-username>/trackx-ai.git
cd trackx-ai
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.simulate           # (re)generate SIMULATED data into data/simulated/
python -m src.classify           # train RF + evaluate on held-out sessions
streamlit run src/dashboard.py   # open the dashboard
```

## How the classifier works

1. **Features** (`src/features.py`): echo amplitude and arrival time, energy, dominant frequency / centroid / bandwidth,
   SNR, attenuation ratio, echo count. Raw waveforms are kept too, for a possible 1D-CNN later.
2. **Random Forest** (`src/classify.py`) predicts normal vs defect probability.
3. **Inconclusive rule:** a reading becomes *Inconclusive* if SNR < 12 dB, no echo is found, or the top class
   probability < 0.80 - instead of forcing a guess. (Thresholds are placeholders to tune on real data.)
4. **Evaluation:** recall, precision, false-alarm rate and share of inconclusive readings, on **unseen sessions**.

## Results so far (simulated only)

On the simulated set the pipeline separates the classes easily (recall/precision ≈ 1.0 on held-out simulated
sessions, and the weak "bad coupling" captures are routed to *Inconclusive*). This only shows the software works.
The simulator is a toy model, so these scores are **expected and say nothing about real rails**.

## Roadmap (indicative)

| Phase | Months | Focus |
|---|---|---|
| 1 | 0-2 | Select probe + pulser-receiver, design ESP32 nodes, plan data collection |
| 2 | 2-5 | Build two nodes, collect labelled waveforms (simulated data only for software) |
| 3 | 5-9 | Feature extraction, train RF / SVM, test on unseen samples |
| 4 | 9-12+ | Dashboard integration, 1D-CNN if data allows, validate against established methods |

## Known limitations

* No real measured data yet; all included data is simulated.
* Firmware is a protocol skeleton. The ESP32 internal ADC is far too slow for MHz ultrasound - a real build needs an
  external high-speed ADC or analog front-end.
* Joints, welds, temperature, vibration and probe coupling change real signals; location-wise data and
  compensation are planned.
* Budget (rough planning range ₹46,000 - ₹1.5 lakh, mid-point about ₹98,000) is an estimate, not vendor quotes.

## References

See `docs/SRS.pdf` (Rose 2014; Rose, Avioli & Song 2002; Cawley et al. 2003; Coccia et al. 2011; Rizzo et al. 2010;
Ministry of Railways, Lok Sabha Q.2, 2009; scikit-learn and Espressif ESP32 / ESP-NOW docs).

## License

MIT - see [LICENSE](LICENSE).
