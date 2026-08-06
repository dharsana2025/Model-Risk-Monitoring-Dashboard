# Model Risk Monitoring Dashboard

I built this because every fraud detection project I've seen ends the same way — someone trains a model, gets a decent AUC score, and calls it done. Nobody shows what happens after deployment. Nobody asks what happens when spending patterns shift, a new fraud ring emerges, or the population the model was trained on starts looking nothing like the population it's scoring today.

This project tries to answer that question.

It trains a LightGBM fraud classifier on synthetic credit card transaction data, then simulates six months of production drift and watches the model gradually degrade. The monitoring system tracks exactly when, where, and how badly it fails — and generates the kind of report you'd actually present to a model risk committee.

This is what model risk teams at banks like AmEx actually do after a model goes live.

---

## The regulatory context (India)

RBI's August 2024 draft circular, "Regulatory Principles for Management of Model Risks in Credit," explicitly requires banks to monitor models on an ongoing basis — checking for population drift, performance degradation, and changing risk patterns. This is not optional and it is not a one-time exercise. The circular draws from the same principles as the US Federal Reserve's SR 11-7 guidance, which shapes how global banks like AmEx govern their internal model risk frameworks.

Most ML projects demonstrate the ability to build a model. This one demonstrates the ability to govern one — which is what RBI and AmEx's MRMG are actually asking for.

---

## What's in here

**PSI (Population Stability Index)** — the core metric for monitoring feature drift. PSI measures how much each feature's distribution has shifted from the training baseline. A PSI above 0.20 means the population the model was trained on no longer looks like the population it's scoring. The dashboard shows this as a heatmap across all features and all months, so you can see at a glance which features are drifting and how fast.

**KS Test (Kolmogorov-Smirnov)** — statistical significance test for drift. Where PSI tells you how much the distribution has changed, the KS test tells you whether that change is statistically real or just noise. A p-value below 0.05 means the shift is significant at 95% confidence.

**Performance tracking** — AUC-ROC, AUC-PR, F1, Precision, and Recall tracked month over month. The dashboard plots these against the baseline and draws a red alert line at the threshold where the degradation becomes a material concern.

**Business impact calculator** — this is the part most ML projects skip. A drop in recall is not just a number — it's missed fraud transactions and actual rupees. If your model's recall drops 8 percentage points and you process 100,000 transactions a day with a 0.5% fraud rate, that's 40 fraudulent transactions per day slipping through. The dashboard multiplies this out using the average fraud transaction amount from the training data.

**Alert engine** — automatically classifies findings as CRITICAL, WARNING, or CLEAR with a recommended action attached to each alert. The logic mirrors what a real monitoring team would escalate: a PSI breach above 0.20 on multiple features triggers revalidation, a performance drop above 5 percentage points triggers model risk committee escalation.

**MRM Report** — a summary report in the format you'd submit to a model risk committee. Covers model overview, baseline performance, monitoring findings, conditions for continued use, and monitoring triggers. Downloadable as JSON for further processing.

---

## Running it

Python 3.9 or higher. No API keys, no Kaggle downloads, no external data.

```bash
cd project3_monitoring
pip install -r requirements.txt
streamlit run app.py
```

Browser opens at `localhost:8501`. Click **Run Full Analysis** in the left sidebar. Takes about 30 seconds to generate all the synthetic data, train the model, and simulate six months of drift.

The sliders let you configure training sample size, baseline fraud rate (real credit card fraud in India runs around 0.1-0.3%), monitoring period, and decision threshold. Play with the fraud rate and threshold sliders — you'll see how they interact with the business impact numbers in interesting ways.

---

## How the drift simulation works

The synthetic data mimics the structure of real credit card fraud datasets — 20 numeric features, a transaction amount column, and a binary fraud label. Legitimate and fraudulent transactions are generated from distinct distributions so the model actually has something real to learn.

The drift is injected in three ways:

First, the first five features shift gradually each month — simulating real-world changes like spending pattern shifts, new merchant categories, or changes in customer demographics. The shift compounds month over month.

Second, the transaction amount distribution inflates each month — simulating economic factors like inflation or seasonal spending changes.

Third, from month three onward, a small number of previously legitimate transactions are relabeled as fraud — simulating a new fraud ring using a technique the model hasn't seen before. This is the part that causes the sharpest performance degradation in the later months.

---

## PSI thresholds

These are industry standard and referenced in both RBI model risk guidance and international frameworks:

- Below 0.10 — population is stable, no action required
- 0.10 to 0.20 — minor shift, worth investigating, increase monitoring frequency
- Above 0.20 — major shift, model may need recalibration or redevelopment, escalate

---

## The model

LightGBM with `scale_pos_weight` to handle class imbalance, early stopping at 60 rounds on validation AUC-PR, and a conservative learning rate of 0.04. Nothing exotic. The model is a vehicle for demonstrating the monitoring system — the interesting engineering is in what happens after it's deployed, not in the model itself.

---

## Tech stack

```
streamlit       — dashboard UI
lightgbm        — gradient boosted tree classifier
scikit-learn    — train/test split and evaluation metrics
plotly          — interactive charts and heatmaps
scipy           — KS test via stats.ks_2samp
numpy / pandas  — data generation and manipulation
```

All of this runs locally. The only thing that touches the network is pip install.

---

## A note on the data

Everything is synthetic. The feature distributions were calibrated to produce realistic model behavior — good baseline AUC, plausible degradation curves, drift patterns that match what you'd see in real production monitoring. But this is not real transaction data and the model should not be used to make any real decisions. It is a demonstration of the monitoring methodology.
