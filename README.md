# Model Risk Monitoring Dashboard

**🔗 Live app: [model-risk-monitoring-dashboard.streamlit.app](https://model-risk-monitoring-dashboard-jz3htxd9tog6kddpysauqx.streamlit.app/)** — opens directly in the browser, no install needed.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://model-risk-monitoring-dashboard-jz3htxd9tog6kddpysauqx.streamlit.app/)

## Objective

Show what a fraud detection model does *after* it goes live — not the AUC score on day one, but how it degrades over six months of real-world drift, and whether the monitoring system catches that degradation before it costs real money.

Every fraud detection tutorial ends the same way: train a model, report an AUC score, done. That's the easy 80%. The harder, more valuable skill — the one RBI's 2023 Draft Circular on Model Risk Management explicitly requires banks to demonstrate — is proving you can watch a deployed model, detect when its world has shifted, and act before performance quietly collapses. This dashboard is that monitoring layer, built and running, not just described.

---

## The problem

A model that scores well on its training data can still fail silently in production. Spending patterns shift. New fraud rings use techniques the model has never seen. The population the model was validated against stops matching the population it's actually scoring. None of that shows up unless something is specifically watching for it — which is what this dashboard does: it trains a LightGBM fraud classifier, then simulates six months of exactly that kind of drift, and tracks precisely when and how the model starts to break.

---

## What's in here

**PSI (Population Stability Index)** — how far each feature's distribution has drifted from the training baseline, shown as a heatmap across every feature and every month. Above 0.20 means the population the model was trained on no longer resembles the one it's scoring.

**KS Test** — a statistical significance check on that drift. PSI tells you *how much* has shifted; the KS test tells you whether that shift is real or just noise (p < 0.05 = real).

**Performance tracking** — AUC-ROC, AUC-PR, F1, Precision, Recall plotted against the baseline, with a red alert line at the point degradation becomes material.

**Business impact calculator** — translates a recall drop into missed fraud transactions and dollar exposure per day, using the actual average fraud amount from the training data.

**Alert engine** — auto-classifies findings as CRITICAL / WARNING / CLEAR with a specific recommended action attached to each.

**MRM Report** — a model-risk-committee-style summary: baseline performance, monitoring findings, conditions for continued use, escalation triggers. Downloadable as JSON.

---

## Running it locally

```bash
cd project3_monitoring
pip install -r requirements.txt
streamlit run app.py
```

Opens at `localhost:8501`. Configure the sidebar, click **Run Full Analysis**. No API keys, no downloads — everything is generated synthetically at runtime.

---

## Deployment

Already live on Streamlit Community Cloud — no install required, just open the link at the top of this README.

To deploy your own copy from a fork:

1. **Push this folder to a public GitHub repo** (if it isn't already there).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **New app** → select your repo, branch `main`, and set the main file path to `app.py`.
4. Click **Deploy**. First build takes 2-3 minutes.
5. You get a permanent link in the shape `https://<your-app-name>.streamlit.app`.

---

## PSI thresholds

Below 0.10 — stable, no action. 0.10 to 0.20 — minor shift, investigate. Above 0.20 — major shift, escalate for recalibration or redevelopment. These are the same thresholds referenced in RBI model risk guidance and international model validation standards.

---

## Tech

```
streamlit       dashboard UI
lightgbm        gradient boosted tree classifier
scikit-learn    train/test split, evaluation metrics
plotly          interactive charts and heatmaps
scipy           KS test (stats.ks_2samp)
numpy / pandas  data generation and manipulation
```

Everything runs locally or on Streamlit Cloud's free tier. No external API, no network call beyond the initial pip install.

---

## A note on the data

Everything is synthetic, calibrated to produce realistic model behavior — a strong but imperfect baseline, plausible drift curves — but this is not real transaction data and shouldn't inform any real decision. It's a demonstration of the monitoring methodology, not a production fraud system.
