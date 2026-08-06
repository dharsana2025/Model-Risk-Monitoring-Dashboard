# 📊 Model Risk Monitoring Dashboard
### Credit Card Fraud Detection · MRMG Framework · SR 11-7 Aligned

---

## What This Does

A production-grade model monitoring system that a bank's Model Risk Management Group (MRMG)
would use to continuously validate a fraud detection model after deployment.

It trains a LightGBM fraud classifier on synthetic credit-card data, then simulates
6 months of real-world production drift — and tracks exactly how and when the model breaks.

---

## Key Features

| Module | What it measures |
|---|---|
| **PSI Heatmap** | Population Stability Index across all features, every month |
| **KS Test** | Kolmogorov-Smirnov statistical drift detection (p < 0.05 threshold) |
| **Performance Tracking** | AUC-ROC, AUC-PR, F1, Precision, Recall degradation curves |
| **Business Impact** | Dollar exposure from missed fraud at each drift level |
| **Alert Engine** | Auto-generated CRITICAL / WARNING / CLEAR alerts with actions |
| **MRM Report** | Downloadable validation report for model risk committee submission |

---

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch dashboard
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## How to Use

1. Configure **training samples**, **fraud rate**, **monitoring months**, and **decision threshold** in the sidebar
2. Click **Run Full Analysis**
3. Explore the 6 tabs:
   - **Overview** — model summary, feature importance, class distribution
   - **Performance** — metric degradation over time + business dollar impact
   - **PSI Analysis** — heatmap + feature drill-down + stability table
   - **Drift Detection** — KS test results + distribution overlays
   - **Alert Center** — severity-ranked alerts with recommended actions
   - **MRM Report** — auto-generated regulatory report (downloadable as JSON)

---

## Technical Details

**Model:** LightGBM with `scale_pos_weight` for class imbalance compensation  
**Data:** Synthetic 20-feature fraud dataset (mimics Kaggle creditcard.csv structure)  
**Drift simulation:** Progressive feature shift (V1-V5) + amount inflation + rising fraud rate  
**PSI formula:** `Σ (Actual% − Expected%) × ln(Actual% / Expected%)`  
**KS test:** `scipy.stats.ks_2samp` at α = 0.05 significance

---

## Regulatory Context

Aligned with:
- **SR 11-7** — Federal Reserve Model Risk Management Guidance
- **OCC 2011-12** — Supervisory Guidance on Model Risk Management
- PSI thresholds per industry standard: < 0.10 stable, 0.10-0.20 investigate, > 0.20 major shift

---

## Portfolio Notes

This project demonstrates the skills MRMG teams look for:
- Independent model validation (not just model building)
- Quantitative stress testing under population shift
- Regulatory vocabulary and framework awareness
- Business impact translation of model degradation
- Production monitoring system design
