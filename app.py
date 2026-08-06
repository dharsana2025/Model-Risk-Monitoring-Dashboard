"""
============================================================
  Model Risk Monitoring Dashboard
  Credit Card Fraud Detection — LightGBM
  MRMG Validation Framework · SR 11-7 Aligned
============================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score
)
from scipy import stats
import warnings
import json
from datetime import datetime
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="MRM Monitoring Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a237e; margin-bottom: 0; }
    .sub-header  { font-size: 0.95rem; color: #546e7a; margin-top: 0; margin-bottom: 1.2rem; }
    div[data-testid="metric-container"] {
        background: #f8f9fa; border-radius: 8px; padding: 12px;
        border-left: 4px solid #1565c0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def generate_baseline(n: int = 15_000, fraud_pct: float = 0.005, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic credit-card transactions that mimic the Kaggle fraud dataset structure.
    V1-V20: PCA-style numeric features | Amount: log-normal | Class: 0=legit, 1=fraud

    Realistic design:
    - Only 5 of 20 features are actually discriminative (like real PCA-transformed fraud data)
    - Feature shifts are 0.8-1.5 sigma apart — enough to learn from, not trivially separable
    - Both classes share similar variance (fraud is not 1.8x noisier)
    - Amount distributions overlap meaningfully
    Target baseline AUC-ROC: ~0.93-0.97 (realistic for a fraud model, not 1.0)
    """
    rng = np.random.default_rng(seed)

    n_fraud = max(int(n * fraud_pct), 50)
    n_legit = n - n_fraud

    # ── Legitimate transactions ──────────────────────────────────
    # All 20 features drawn from N(0,1); only 5 get a small class-level shift
    L = rng.standard_normal((n_legit, 20))
    L[:, 0] -= 0.4   # V1: legit slightly negative
    L[:, 1] += 0.6   # V2: legit slightly positive
    L[:, 2] -= 0.5   # V3
    L[:, 3] += 0.7   # V4
    L[:, 4] -= 0.3   # V5
    # V6-V20: pure noise, no class signal
    amt_L = rng.lognormal(4.2, 1.1, n_legit)   # typical transaction amounts

    # ── Fraud transactions ───────────────────────────────────────
    # Same variance as legit (std ≈ 1.0-1.1), moderate shifts on same 5 features
    # This creates real overlap — model must learn, not just threshold
    F = rng.standard_normal((n_fraud, 20)) * 1.1   # slightly higher noise
    F[:, 0] += 1.4   # V1: fraud pulls in opposite direction (total gap ~1.8)
    F[:, 1] -= 1.8   # V2: total gap ~2.4 — most discriminative feature
    F[:, 2] += 1.2   # V3: total gap ~1.7
    F[:, 3] -= 1.5   # V4: total gap ~2.2
    F[:, 4] += 0.9   # V5: total gap ~1.2 — weakest signal
    # V6-V20: also pure noise for fraud
    amt_F = rng.lognormal(4.5, 1.3, n_fraud)   # slightly higher amounts, but overlapping

    X   = np.vstack([L, F])
    amt = np.concatenate([amt_L, amt_F])
    y   = np.array([0] * n_legit + [1] * n_fraud)

    cols = [f"V{i}" for i in range(1, 21)]
    df = pd.DataFrame(X, columns=cols)
    df["Amount"] = np.abs(amt)
    df["Class"]  = y
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def drift_data(base_df: pd.DataFrame, month: int) -> pd.DataFrame:
    """
    Inject progressive concept drift into baseline data.
    Simulates real-world population shift over time:
    - Feature distributions shift each month
    - Amount inflation (e.g. seasonal spending)
    - Fraud rate rises from month 3 (new attack patterns)
    """
    rng = np.random.default_rng(month * 137)
    df  = base_df.copy()
    n   = len(df)
    d   = month * 0.18   # drift strength grows each month

    shift_map = {"V1": d * 0.9, "V2": d * 1.3, "V3": d * 0.6,
                 "V4": d * 1.0, "V5": d * 0.4}
    for feat, shift in shift_map.items():
        df[feat] += rng.normal(shift, abs(shift) * 0.15, n)

    df["Amount"] *= (1 + 0.06 * month)          # inflation

    if month >= 3:                               # new fraud ring emerges
        extra = int(0.0015 * month * n)
        pool  = df.index[df["Class"] == 0].tolist()
        if len(pool) > extra:
            df.loc[rng.choice(pool, extra, replace=False), "Class"] = 1

    return df


# ─────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_model(train_df: pd.DataFrame):
    """Train LightGBM fraud classifier; return model, feature list, baseline metrics."""
    feats = [c for c in train_df.columns if c != "Class"]
    X, y  = train_df[feats], train_df["Class"]

    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pos_w = len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1)

    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.04,
        max_depth=6, num_leaves=63,
        scale_pos_weight=pos_w,
        class_weight="balanced",
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)]
    )

    p_vl = clf.predict_proba(X_vl)[:, 1]
    base_metrics = {
        "AUC-ROC": round(roc_auc_score(y_vl, p_vl), 4),
        "AUC-PR" : round(average_precision_score(y_vl, p_vl), 4),
        "F1"     : round(f1_score(y_vl, p_vl >= 0.5, zero_division=0), 4),
        "Prec"   : round(precision_score(y_vl, p_vl >= 0.5, zero_division=0), 4),
        "Recall" : round(recall_score(y_vl, p_vl >= 0.5), 4),
    }
    return clf, feats, base_metrics


def score_df(clf, df: pd.DataFrame, feats: list, thr: float = 0.5) -> tuple:
    """Score a dataset and return metrics + probability array."""
    X, y = df[feats], df["Class"]
    p    = clf.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    metrics = {
        "AUC-ROC" : round(roc_auc_score(y, p), 4)              if y.nunique() > 1 else None,
        "AUC-PR"  : round(average_precision_score(y, p), 4)     if y.nunique() > 1 else None,
        "F1"      : round(f1_score(y, pred, zero_division=0), 4),
        "Prec"    : round(precision_score(y, pred, zero_division=0), 4),
        "Recall"  : round(recall_score(y, pred), 4),
        "Fraud_%" : round(y.mean() * 100, 3),
        "Pred_%"  : round(pred.mean() * 100, 3),
    }
    return metrics, p


# ─────────────────────────────────────────
# PSI / KS
# ─────────────────────────────────────────

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)
    Thresholds: < 0.10 Stable | 0.10-0.20 Investigate | > 0.20 Major shift
    """
    bpts = np.unique(np.nanpercentile(expected, np.linspace(0, 100, bins + 1)))
    if len(bpts) < 2:
        return 0.0
    eps = 1e-6
    e_c = np.histogram(expected, bpts)[0] + eps
    a_c = np.histogram(actual,   bpts)[0] + eps
    e_p, a_p = e_c / e_c.sum(), a_c / a_c.sum()
    return round(float(np.abs(np.sum((a_p - e_p) * np.log(a_p / e_p)))), 4)


def all_psi(base_df: pd.DataFrame, prod_df: pd.DataFrame, feats: list) -> dict:
    return {f: psi(base_df[f].values, prod_df[f].values) for f in feats}


def ks_test(base_s: pd.Series, prod_s: pd.Series, alpha: float = 0.05) -> tuple:
    stat, p = stats.ks_2samp(base_s, prod_s)
    return round(stat, 4), round(p, 6), bool(p < alpha)


def psi_label(v: float) -> str:
    if v < 0.10: return "✅ Stable"
    if v < 0.20: return "⚠️ Investigate"
    return "🚨 Major Shift"


# ─────────────────────────────────────────
# ALERT ENGINE
# ─────────────────────────────────────────

def gen_alerts(base_m: dict, curr_m: dict, psi_dict: dict,
               curr_df: pd.DataFrame, base_fraud_pct: float) -> list:
    alerts = []

    # Performance degradation
    if curr_m.get("AUC-ROC"):
        drop = base_m["AUC-ROC"] - curr_m["AUC-ROC"]
        if drop >= 0.05:
            alerts.append(("🔴 CRITICAL", "Performance",
                f"AUC-ROC dropped {drop:.4f} below baseline ({base_m['AUC-ROC']} → {curr_m['AUC-ROC']})",
                "Immediate revalidation; escalate to Model Risk Committee"))
        elif drop >= 0.02:
            alerts.append(("🟡 WARNING", "Performance",
                f"AUC-ROC declining — {drop:.4f} below baseline",
                "Schedule model review within 30 days"))

    # Recall degradation
    if curr_m.get("Recall") is not None:
        r_drop = base_m["Recall"] - curr_m["Recall"]
        if r_drop >= 0.10:
            alerts.append(("🔴 CRITICAL", "Performance",
                f"Recall dropped {r_drop:.4f} below baseline ({base_m['Recall']} → {curr_m['Recall']}) — rising missed-fraud rate",
                "Emergency model review; escalate to Model Risk Committee"))
        elif r_drop >= 0.05:
            alerts.append(("🟡 WARNING", "Performance",
                f"Recall declining — {r_drop:.4f} below baseline",
                "Schedule model review within 30 days"))

    # PSI breaches
    crit_feats = [f for f, v in psi_dict.items() if v >= 0.20]
    warn_feats  = [f for f, v in psi_dict.items() if 0.10 <= v < 0.20]
    if crit_feats:
        alerts.append(("🔴 CRITICAL", "Population Drift",
            f"PSI ≥ 0.20 in {len(crit_feats)} feature(s): {', '.join(crit_feats[:5])}",
            "Model recalibration or redevelopment required"))
    if warn_feats:
        alerts.append(("🟡 WARNING", "Population Drift",
            f"PSI 0.10-0.20 in {len(warn_feats)} feature(s)",
            "Increase monitoring frequency; assess recalibration need"))

    # Fraud rate surge
    live_fr = curr_df["Class"].mean()
    if live_fr > base_fraud_pct * 2:
        alerts.append(("🔴 CRITICAL", "Fraud Pattern Shift",
            f"Fraud rate doubled: {live_fr*100:.3f}% vs baseline {base_fraud_pct*100:.3f}%",
            "Alert fraud operations; convene emergency model review"))

    if not alerts:
        alerts.append(("🟢 CLEAR", "System",
            "No material issues detected across all monitoring dimensions",
            "Continue standard monthly monitoring"))
    return alerts


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

def main():
    st.markdown('<p class="main-header">🔍 Model Risk Monitoring Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Credit Card Fraud Detection · LightGBM · '
        'MRMG Validation Framework · SR 11-7 Aligned</p>', unsafe_allow_html=True
    )

    # ── Sidebar ──────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        n_samp   = st.slider("Training Samples",       5_000, 25_000, 15_000, 1_000)
        fr_pct   = st.slider("Baseline Fraud Rate (%)", 0.1,   2.0,   0.5,    0.1) / 100
        n_months = st.slider("Monitoring Months",       3,     6,     6)
        thr      = st.slider("Decision Threshold",      0.1,   0.9,   0.5,    0.05)

        st.markdown("---")
        st.markdown("### 📐 PSI Thresholds")
        st.markdown("🟢 **< 0.10** — Stable")
        st.markdown("🟡 **0.10 – 0.20** — Investigate")
        st.markdown("🔴 **> 0.20** — Major shift")
        st.markdown("---")
        run = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    tabs = st.tabs([
        "🏠 Overview", "📈 Performance",
        "📊 PSI Analysis", "🔬 Drift Detection",
        "⚠️ Alert Center", "📋 MRM Report"
    ])

    # Welcome screen
    if not run and "pipeline_done" not in st.session_state:
        with tabs[0]:
            st.info("👈 Configure settings in the sidebar and click **Run Full Analysis** to start.")
            c1, c2, c3 = st.columns(3)
            c1.markdown("""
**What this monitors**
- Feature distribution shift (PSI)
- Statistical drift (KS test)
- AUC / F1 / Precision / Recall
- Fraud-rate surge detection
""")
            c2.markdown("""
**Key outputs**
- Monthly PSI heatmap
- KS drift significance table
- Performance degradation curves
- Auto-generated MRM alerts
""")
            c3.markdown("""
**Regulatory alignment**
- SR 11-7 MRM guidance
- OCC 2011-12 validation standard
- Model risk committee report
- Conditions for continued use
""")
        return

    # ── Run pipeline ─────────────────────
    with st.spinner("Generating baseline data…"):
        base_df = generate_baseline(n_samp, fr_pct)

    with st.spinner("Training LightGBM fraud model…"):
        clf, feats, base_m = train_model(base_df)

    monthly, m_metrics, m_psi, m_ks = {}, {}, {}, {}
    prog = st.progress(0, text="Simulating production months…")
    for mo in range(1, n_months + 1):
        prod              = drift_data(base_df, mo)
        metrics, _        = score_df(clf, prod, feats, thr)
        psi_res           = all_psi(base_df, prod, feats)
        ks_res            = {f: ks_test(base_df[f], prod[f])
                              for f in ["V1", "V2", "V3", "Amount"] if f in feats}
        monthly[mo]    = prod
        m_metrics[mo]  = metrics
        m_psi[mo]      = psi_res
        m_ks[mo]       = ks_res
        prog.progress(mo / n_months, text=f"Simulated month {mo} of {n_months}…")
    prog.empty()
    st.session_state["pipeline_done"] = True

    months = list(range(1, n_months + 1))

    # ══════════════════════════════════════
    # TAB 0 — OVERVIEW
    # ══════════════════════════════════════
    with tabs[0]:
        st.subheader("Baseline Model Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("AUC-ROC",   base_m["AUC-ROC"])
        c2.metric("AUC-PR",    base_m["AUC-PR"])
        c3.metric("F1 Score",  base_m["F1"])
        c4.metric("Recall",    base_m["Recall"])
        c5.metric("Precision", base_m["Prec"])

        st.markdown("---")
        ca, cb = st.columns(2)

        with ca:
            fi_df = (pd.DataFrame({"Feature": feats, "Gain": clf.feature_importances_})
                     .sort_values("Gain", ascending=False).head(15))
            fig = px.bar(
                fi_df, x="Gain", y="Feature", orientation="h",
                color="Gain", color_continuous_scale="Blues",
                title="Top 15 Features by Gain (LightGBM)"
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"},
                              height=430, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with cb:
            cd = base_df["Class"].value_counts()
            fig = px.pie(
                values=cd.values, names=["Legitimate", "Fraud"],
                color_discrete_sequence=["#1565c0", "#c62828"],
                title=f"Class Distribution — Fraud Rate: {fr_pct*100:.2f}%",
                hole=0.35
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Transaction Amount Distribution by Class")
        samp = base_df.sample(min(3000, len(base_df)), random_state=1)
        fig = px.histogram(
            samp, x="Amount", color="Class", nbins=60,
            barmode="overlay", opacity=0.7,
            color_discrete_sequence=["#1565c0", "#c62828"],
            labels={"Class": "0=Legit / 1=Fraud"},
            title="Legitimate vs Fraud — Amount Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════
    # TAB 1 — PERFORMANCE
    # ══════════════════════════════════════
    with tabs[1]:
        st.subheader("Model Performance Degradation Over Time")

        x_labels = ["Baseline"] + [f"M{m}" for m in months]
        metric_keys = ["AUC-ROC", "AUC-PR", "F1", "Recall"]
        colors      = ["#1565c0", "#2e7d32", "#e65100", "#b71c1c"]

        fig = make_subplots(rows=2, cols=2, subplot_titles=metric_keys)
        for idx, (mk, col) in enumerate(zip(metric_keys, colors)):
            row, c = divmod(idx, 2)
            ys = [base_m.get(mk)] + [m_metrics[m].get(mk) for m in months]
            pairs = [(x, y) for x, y in zip(x_labels, ys) if y is not None]
            if pairs:
                vx, vy = zip(*pairs)
                fig.add_trace(
                    go.Scatter(x=list(vx), y=list(vy), mode="lines+markers",
                               name=mk, line=dict(color=col, width=2.5),
                               marker=dict(size=9)),
                    row=row + 1, col=c + 1
                )
            if mk == "AUC-ROC":
                fig.add_hline(
                    y=round(base_m["AUC-ROC"] - 0.05, 4),
                    line_dash="dash", line_color="red",
                    annotation_text="Alert (-5pp)", row=1, col=1
                )
        fig.update_layout(height=500, showlegend=False,
                          title_text="Key Performance Indicators — Baseline → Production")
        st.plotly_chart(fig, use_container_width=True)

        # Metrics table with conditional formatting
        rows = [{"Period": "Baseline", **base_m, "Fraud_%": round(fr_pct * 100, 3)}]
        rows += [{"Period": f"M{m}", **m_metrics[m]} for m in months]
        perf_tbl = pd.DataFrame(rows)

        def _fmt(v):
            if isinstance(v, float): return f"{v:.4f}"
            return v

        st.dataframe(
            perf_tbl[["Period", "AUC-ROC", "AUC-PR", "F1", "Prec", "Recall", "Fraud_%"]]
            .style.format({k: "{:.4f}" for k in ["AUC-ROC","AUC-PR","F1","Prec","Recall"]},
                          na_rep="—"),
            use_container_width=True
        )

        # Business impact
        st.markdown("---")
        st.subheader("💰 Business Impact Estimation  *(100K daily transactions)*")
        last   = months[-1]
        r_drop = base_m["Recall"] - m_metrics[last]["Recall"]
        avg_a  = base_df[base_df["Class"] == 1]["Amount"].mean()
        missed = max(0, r_drop) * 100_000 * fr_pct
        loss   = missed * avg_a

        c1, c2, c3 = st.columns(3)
        c1.metric("Recall Drop (Baseline → Final)", f"{r_drop:+.2%}", delta_color="inverse")
        c2.metric("Missed Fraud Transactions / Day", f"{missed:,.0f}")
        c3.metric("Estimated Daily Exposure",         f"${loss:,.0f}")

    # ══════════════════════════════════════
    # TAB 2 — PSI
    # ══════════════════════════════════════
    with tabs[2]:
        st.subheader("Population Stability Index (PSI)")
        st.caption("PSI < 0.10 Stable · 0.10-0.20 Investigate · > 0.20 Model validity at risk")

        psi_mat = pd.DataFrame(m_psi).T
        psi_mat.index = [f"Month {m}" for m in psi_mat.index]
        top_feats = psi_mat.mean().sort_values(ascending=False).head(15).index
        psi_disp  = psi_mat[top_feats]

        fig = px.imshow(
            psi_disp,
            color_continuous_scale=[
                [0.0,  "#1b5e20"], [0.10, "#66bb6a"],
                [0.17, "#ffd600"], [0.22, "#e53935"], [1.0,  "#4a0000"]
            ],
            zmin=0, zmax=0.35, text_auto=".3f", aspect="auto",
            title="PSI Heatmap — Top 15 Features (Green=Stable · Red=Drifted)"
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Feature-Level PSI Trend")
        sel = st.selectbox("Select feature", feats, key="psi_feat")
        trend  = [m_psi[m].get(sel, 0) for m in months]
        c_list = ["#1b5e20" if v < 0.10 else "#ffd600" if v < 0.20 else "#e53935" for v in trend]

        fig2 = go.Figure(go.Bar(
            x=[f"M{m}" for m in months], y=trend,
            marker_color=c_list,
            text=[f"{v:.4f}" for v in trend], textposition="outside"
        ))
        fig2.add_hline(y=0.10, line_dash="dot", line_color="orange", annotation_text="0.10")
        fig2.add_hline(y=0.20, line_dash="dot", line_color="red",    annotation_text="0.20")
        fig2.update_layout(title=f"PSI Over Time — {sel}", yaxis_title="PSI", height=320)
        st.plotly_chart(fig2, use_container_width=True)

        # Summary table
        final_psi = m_psi[months[-1]]
        psi_tbl = pd.DataFrame([
            {"Feature": k, "PSI": v, "Status": psi_label(v)}
            for k, v in sorted(final_psi.items(), key=lambda x: -x[1])
        ])
        st.markdown(f"#### PSI Summary — Month {months[-1]} (Final)")
        st.dataframe(psi_tbl.style.format({"PSI": "{:.4f}"}), use_container_width=True)

    # ══════════════════════════════════════
    # TAB 3 — DRIFT
    # ══════════════════════════════════════
    with tabs[3]:
        st.subheader("Statistical Drift Detection — Kolmogorov-Smirnov Test")
        st.caption("p-value < 0.05 → distribution shift is statistically significant at 95% confidence")

        ks_feats = list(m_ks[1].keys()) if m_ks else []
        if ks_feats:
            ks_mat = pd.DataFrame({
                f"M{m}": {f: m_ks[m][f][0] for f in ks_feats} for m in months
            }).T
            fig = px.imshow(
                ks_mat, color_continuous_scale="RdYlGn_r",
                text_auto=".3f", aspect="auto",
                title="KS Statistic Heatmap — Higher Value = Greater Drift"
            )
            fig.update_layout(height=260)
            st.plotly_chart(fig, use_container_width=True)

            # KS table
            ks_rows = []
            for feat in ks_feats:
                stat, pval, drifted = m_ks[months[-1]][feat]
                ks_rows.append({
                    "Feature": feat, "KS Statistic": stat,
                    "p-value": pval,
                    "Drift Detected": "🚨 Yes" if drifted else "✅ No"
                })
            st.markdown(f"#### KS Test Results — Month {months[-1]}")
            st.dataframe(pd.DataFrame(ks_rows), use_container_width=True)

        # Distribution overlays
        st.markdown("#### Distribution Comparison: Baseline vs Final Month")
        last_prod = monthly[months[-1]]
        c1, c2 = st.columns(2)
        for idx, feat in enumerate(["V1", "V2", "V3", "Amount"]):
            if feat not in base_df.columns: continue
            lo = base_df[feat].quantile(0.01)
            hi = base_df[feat].quantile(0.99)
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=base_df[feat].clip(lo, hi), name="Baseline",
                opacity=0.6, marker_color="#1565c0", nbinsx=50
            ))
            fig.add_trace(go.Histogram(
                x=last_prod[feat].clip(lo, hi), name=f"Month {months[-1]}",
                opacity=0.6, marker_color="#c62828", nbinsx=50
            ))
            ks_info = m_ks.get(months[-1], {}).get(feat, (None, None, None))
            title = f"{feat}"
            if ks_info[0]:
                title += f"  |  KS={ks_info[0]:.3f}  p={ks_info[1]:.4f}"
            fig.update_layout(barmode="overlay", title=title,
                              height=270, legend=dict(orientation="h", y=1.15))
            with (c1 if idx % 2 == 0 else c2):
                st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════
    # TAB 4 — ALERTS
    # ══════════════════════════════════════
    with tabs[4]:
        st.subheader("⚠️ Model Alert Center")
        last   = months[-1]
        alerts = gen_alerts(base_m, m_metrics[last], m_psi[last], monthly[last], fr_pct)

        bg_map = {
            "🔴 CRITICAL": "#ffcdd2",
            "🟡 WARNING" : "#fff9c4",
            "🟢 CLEAR"   : "#c8e6c9"
        }
        for sev, cat, desc, action in alerts:
            bg = bg_map.get(sev, "#f5f5f5")
            st.markdown(f"""
<div style="background:{bg};padding:14px;border-radius:8px;margin:8px 0;
            border-left:5px solid {'#c62828' if 'CRITICAL' in sev else '#f57f17' if 'WARNING' in sev else '#2e7d32'}">
  <strong>{sev}</strong> &nbsp;|&nbsp; <em>{cat}</em><br>
  📌 {desc}<br>
  🔧 <strong>Recommended Action:</strong> {action}
</div>""", unsafe_allow_html=True)

        crits  = sum(1 for a in alerts if "🔴" in a[0])
        warns  = sum(1 for a in alerts if "🟡" in a[0])
        status = "🔴 Action Required" if crits else "🟡 Monitor Closely" if warns else "🟢 Stable"

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Critical Alerts", crits,  delta_color="inverse")
        c2.metric("Warnings",        warns,  delta_color="inverse")
        c3.metric("Overall Status",  status)

    # ══════════════════════════════════════
    # TAB 5 — MRM REPORT
    # ══════════════════════════════════════
    with tabs[5]:
        st.subheader("📋 Model Risk Management Report")
        st.caption("Auto-generated · SR 11-7 Aligned · For Model Risk Committee Submission")

        last       = months[-1]
        alerts     = gen_alerts(base_m, m_metrics[last], m_psi[last], monthly[last], fr_pct)
        crit_feats = [f for f, v in m_psi[last].items() if v >= 0.20]
        fin_auc    = m_metrics[last]["AUC-ROC"] or 0
        auc_delta  = round(base_m["AUC-ROC"] - fin_auc, 4)
        rpt_date   = datetime.now().strftime("%d %B %Y")
        has_crit   = any("🔴" in a[0] for a in alerts)
        has_warn   = any("🟡" in a[0] for a in alerts)
        status_str = ("⛔ Requires Immediate Action"   if has_crit else
                      "⚠️ Under Enhanced Monitoring"   if has_warn else
                      "✅ Valid for Continued Use")

        max_psi    = max(m_psi[last].values())
        live_fr    = monthly[last]["Class"].mean()
        fr_ratio   = live_fr / fr_pct

        st.markdown(f"""
---
**Model ID:** FRAUD-LGB-001 &emsp;|&emsp;
**Algorithm:** LightGBM Gradient Boosted Trees &emsp;|&emsp;
**Report Date:** {rpt_date}

**Validation Period:** {n_months} months &emsp;|&emsp;
**Decision Threshold:** {thr} &emsp;|&emsp;
**Model Status:** {status_str}

---
### 1. Model Overview
A gradient-boosted tree classifier (LightGBM) trained to estimate the probability that a
credit-card transaction is fraudulent. The model consumes **{len(feats)} input features**
and outputs a continuous risk score [0,1]. Transactions with score ≥ **{thr}** are routed
to the fraud review queue.

---
### 2. Baseline Performance (Hold-out Validation)

| Metric | Value | Interpretation |
|---|---|---|
| AUC-ROC | {base_m["AUC-ROC"]} | Model rank-orders fraud well above chance |
| AUC-PR  | {base_m["AUC-PR"]}  | Strong precision-recall tradeoff under class imbalance |
| F1 Score | {base_m["F1"]}    | Balanced precision/recall at threshold {thr} |
| Precision | {base_m["Prec"]} | Of flagged transactions, this share are true fraud |
| Recall   | {base_m["Recall"]} | Model captures this share of all fraud cases |

---
### 3. Monitoring Findings — Month {last} (Final Period)

| Dimension | Finding |
|---|---|
| Population Stability | {len(crit_feats)} feature(s) with PSI ≥ 0.20: {', '.join(crit_feats) if crit_feats else 'None'} |
| Performance Trend | AUC-ROC: {base_m["AUC-ROC"]} → {fin_auc} (Δ = {auc_delta:+.4f}) |
| Fraud Rate | {live_fr*100:.3f}% vs baseline {fr_pct*100:.3f}% ({fr_ratio:.2f}× ratio) |

---
### 4. Risk Findings & Recommended Actions
""")
        for i, (sev, cat, desc, action) in enumerate(alerts, 1):
            st.markdown(f"**Finding {i}** [{sev}] *{cat}*  \n📌 {desc}  \n🔧 {action}\n")

        st.markdown(f"""
---
### 5. Conditions for Continued Use

| Condition | Threshold | Current Value | Status |
|---|---|---|---|
| Max feature PSI | < 0.20 | {max_psi:.4f} | {"🚨 Breached" if max_psi >= 0.20 else "✅ OK"} |
| AUC-ROC decline | < 5 pp | {auc_delta:+.4f} | {"🚨 Breached" if auc_delta >= 0.05 else "✅ OK"} |
| Fraud rate ratio | < 2× baseline | {fr_ratio:.2f}× | {"🚨 Breached" if fr_ratio > 2 else "✅ OK"} |
| Recall degradation | < 10 pp | {base_m["Recall"] - m_metrics[last]["Recall"]:+.4f} | {"🚨 Breached" if base_m["Recall"] - m_metrics[last]["Recall"] >= 0.10 else "✅ OK"} |

---
### 6. Monitoring Triggers & Escalation Protocol

| Trigger | Threshold | Frequency | Escalation |
|---|---|---|---|
| Feature PSI breach | ≥ 0.20 any feature | Monthly | Revalidation required within 60 days |
| AUC-ROC decline | > 5 pp absolute | Monthly | Escalate to Model Risk Committee |
| Fraud rate surge | > 2× baseline | Weekly | Alert fraud operations + MRC |
| Recall degradation | > 10 pp | Monthly | Emergency model review |
| Adversarial attack detected | N/A | Real-time | Fraud team + CISO |

---
*Report generated automatically by the MRMG Monitoring System.*  
*Reviewed under SR 11-7 / OCC 2011-12 Model Risk Management framework.*  
*Next scheduled validation: {n_months} months from model deployment.*
""")

        # JSON export
        export = {
            "report_date"    : rpt_date,
            "model_id"       : "FRAUD-LGB-001",
            "monitoring_months": n_months,
            "baseline_metrics": base_m,
            "monthly_metrics" : {str(k): v for k, v in m_metrics.items()},
            "final_psi"       : {k: float(v) for k, v in m_psi[last].items()},
            "alerts"          : [
                {"severity": a[0], "category": a[1],
                 "description": a[2], "action": a[3]}
                for a in alerts
            ],
        }
        st.download_button(
            "📥 Download Full MRM Report (JSON)",
            data=json.dumps(export, indent=2, default=str),
            file_name=f"mrm_report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
