"""
============================================================
  Model Risk Monitoring Dashboard
  Credit Card Fraud Detection — LightGBM
  MRMG Validation Framework · SR 11-7 / RBI Model Risk Aligned
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

st.set_page_config(page_title="MRM Monitoring Dashboard", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a237e; margin-bottom: 0; }
    .sub-header  { font-size: 0.95rem; color: #546e7a; margin-top: 0; margin-bottom: 1.2rem; }
    div[data-testid="metric-container"] {
        background: #f8f9fa; border-radius: 8px; padding: 12px; border-left: 4px solid #1565c0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────

@st.cache_data(show_spinner=False)
def generate_baseline(n: int = 15_000, fraud_pct: float = 0.005, seed: int = 42) -> pd.DataFrame:
    """
    20-feature synthetic transactions mimicking the Kaggle fraud dataset shape.
    Only 5 of 20 features carry class signal, with realistic overlap (0.8-2.4 sigma
    gaps) — a model has to learn, not just threshold. Minimum 200 fraud rows are
    generated regardless of fraud_pct so the validation fold has enough positives
    for AUC to be a meaningful (not sample-noise-driven) statistic.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(n * fraud_pct), 200)
    n_legit = n - n_fraud

    L = rng.standard_normal((n_legit, 20))
    L[:, :5] += [-0.4, 0.6, -0.5, 0.7, -0.3]
    amt_L = rng.lognormal(4.2, 1.1, n_legit)

    F = rng.standard_normal((n_fraud, 20)) * 1.1
    F[:, :5] += [1.4, -1.8, 1.2, -1.5, 0.9]
    amt_F = rng.lognormal(4.5, 1.3, n_fraud)

    X   = np.vstack([L, F])
    amt = np.abs(np.concatenate([amt_L, amt_F]))
    y   = np.array([0] * n_legit + [1] * n_fraud)

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 21)])
    df["Amount"], df["Class"] = amt, y
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def drift_data(base_df: pd.DataFrame, month: int) -> pd.DataFrame:
    """Progressive drift: feature shift + amount inflation + new fraud pattern from month 3."""
    rng = np.random.default_rng(month * 137)
    df, n, d = base_df.copy(), len(base_df), month * 0.18

    for feat, mult in zip(["V1", "V2", "V3", "V4", "V5"], [0.9, 1.3, 0.6, 1.0, 0.4]):
        shift = d * mult
        df[feat] += rng.normal(shift, abs(shift) * 0.15, n)
    df["Amount"] *= (1 + 0.06 * month)

    if month >= 3:
        extra = int(0.0015 * month * n)
        pool  = df.index[df["Class"] == 0].tolist()
        if len(pool) > extra:
            df.loc[rng.choice(pool, extra, replace=False), "Class"] = 1
    return df


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_model(train_df: pd.DataFrame):
    """Train LightGBM; return the model, feature list, and baseline hold-out metrics."""
    feats = [c for c in train_df.columns if c != "Class"]
    X, y = train_df[feats], train_df["Class"]
    X_tr, X_vl, y_tr, y_vl = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = lgb.LGBMClassifier(
        n_estimators=250, learning_rate=0.05, max_depth=6, num_leaves=63,
        scale_pos_weight=len(y_tr[y_tr == 0]) / max(len(y_tr[y_tr == 1]), 1),
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])

    p = clf.predict_proba(X_vl)[:, 1]
    metrics = {
        "AUC-ROC": round(roc_auc_score(y_vl, p), 4),
        "AUC-PR" : round(average_precision_score(y_vl, p), 4),
        "F1"     : round(f1_score(y_vl, p >= 0.5, zero_division=0), 4),
        "Prec"   : round(precision_score(y_vl, p >= 0.5, zero_division=0), 4),
        "Recall" : round(recall_score(y_vl, p >= 0.5), 4),
    }
    return clf, feats, metrics


def score_df(clf, df: pd.DataFrame, feats: list, thr: float = 0.5) -> tuple:
    X, y = df[feats], df["Class"]
    p = clf.predict_proba(X)[:, 1]
    pred = (p >= thr).astype(int)
    multi = y.nunique() > 1
    return {
        "AUC-ROC": round(roc_auc_score(y, p), 4) if multi else None,
        "AUC-PR" : round(average_precision_score(y, p), 4) if multi else None,
        "F1"     : round(f1_score(y, pred, zero_division=0), 4),
        "Prec"   : round(precision_score(y, pred, zero_division=0), 4),
        "Recall" : round(recall_score(y, pred), 4),
        "Fraud_%": round(y.mean() * 100, 3),
    }, p


# ─────────────────────────────────────────
# PSI / KS
# ─────────────────────────────────────────

def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%). <0.10 stable, >0.20 major shift."""
    bpts = np.unique(np.nanpercentile(expected, np.linspace(0, 100, bins + 1)))
    if len(bpts) < 2:
        return 0.0
    eps = 1e-6
    e_p = (np.histogram(expected, bpts)[0] + eps); e_p /= e_p.sum()
    a_p = (np.histogram(actual, bpts)[0] + eps); a_p /= a_p.sum()
    return round(float(np.abs(np.sum((a_p - e_p) * np.log(a_p / e_p)))), 4)


def all_psi(base_df, prod_df, feats) -> dict:
    return {f: psi(base_df[f].values, prod_df[f].values) for f in feats}


def ks_test(base_s, prod_s, alpha=0.05) -> tuple:
    stat, p = stats.ks_2samp(base_s, prod_s)
    return round(stat, 4), round(p, 6), bool(p < alpha)


def psi_label(v: float) -> str:
    return "✅ Stable" if v < 0.10 else "⚠️ Investigate" if v < 0.20 else "🚨 Major Shift"


# ─────────────────────────────────────────
# ALERTS
# ─────────────────────────────────────────

def gen_alerts(base_m, curr_m, psi_dict, curr_df, base_fraud_pct) -> list:
    alerts = []
    if curr_m.get("AUC-ROC"):
        drop = base_m["AUC-ROC"] - curr_m["AUC-ROC"]
        if drop >= 0.05:
            alerts.append(("🔴 CRITICAL", "Performance",
                f"AUC-ROC dropped {drop:.4f} below baseline ({base_m['AUC-ROC']} → {curr_m['AUC-ROC']})",
                "Immediate revalidation; escalate to Model Risk Committee"))
        elif drop >= 0.02:
            alerts.append(("🟡 WARNING", "Performance",
                f"AUC-ROC declining — {drop:.4f} below baseline", "Schedule review within 30 days"))

    crit = [f for f, v in psi_dict.items() if v >= 0.20]
    warn = [f for f, v in psi_dict.items() if 0.10 <= v < 0.20]
    if crit:
        alerts.append(("🔴 CRITICAL", "Population Drift",
            f"PSI ≥ 0.20 in {len(crit)} feature(s): {', '.join(crit[:5])}",
            "Recalibration or redevelopment required"))
    if warn:
        alerts.append(("🟡 WARNING", "Population Drift",
            f"PSI 0.10-0.20 in {len(warn)} feature(s)", "Increase monitoring frequency"))

    live_fr = curr_df["Class"].mean()
    if live_fr > base_fraud_pct * 2:
        alerts.append(("🔴 CRITICAL", "Fraud Pattern Shift",
            f"Fraud rate doubled: {live_fr*100:.3f}% vs baseline {base_fraud_pct*100:.3f}%",
            "Alert fraud ops; convene emergency review"))

    return alerts or [("🟢 CLEAR", "System", "No material issues detected", "Continue standard monitoring")]


# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────

def main():
    st.markdown('<p class="main-header">🔍 Model Risk Monitoring Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Credit Card Fraud Detection · LightGBM · '
                'MRMG Framework · SR 11-7 / RBI Model Risk Aligned</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        n_samp   = st.slider("Training Samples", 5_000, 25_000, 15_000, 1_000)
        fr_pct   = st.slider("Baseline Fraud Rate (%)", 0.1, 2.0, 0.5, 0.1) / 100
        n_months = st.slider("Monitoring Months", 3, 6, 6)
        thr      = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
        st.markdown("---\n### 📐 PSI Thresholds\n🟢 **< 0.10** Stable  \n🟡 **0.10-0.20** Investigate  \n🔴 **> 0.20** Major shift")
        st.markdown("---")
        run = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

    tabs = st.tabs(["🏠 Overview", "📈 Performance", "📊 PSI Analysis",
                     "🔬 Drift Detection", "⚠️ Alert Center", "📋 MRM Report"])

    if not run and "pipeline_done" not in st.session_state:
        with tabs[0]:
            st.info("👈 Configure settings in the sidebar and click **Run Full Analysis** to start.")
            c1, c2, c3 = st.columns(3)
            c1.markdown("**What this monitors**\n- Feature drift (PSI)\n- Statistical drift (KS)\n"
                        "- AUC / F1 / Precision / Recall\n- Fraud-rate surge")
            c2.markdown("**Key outputs**\n- Monthly PSI heatmap\n- KS drift table\n"
                        "- Degradation curves\n- Auto-generated alerts")
            c3.markdown("**Regulatory alignment**\n- SR 11-7 / RBI Model Risk\n- OCC 2011-12\n"
                        "- MRC-style report\n- Conditions for continued use")
        return

    with st.spinner("Generating baseline data…"):
        base_df = generate_baseline(n_samp, fr_pct)
    with st.spinner("Training LightGBM fraud model…"):
        clf, feats, base_m = train_model(base_df)

    months = list(range(1, n_months + 1))
    monthly, m_metrics, m_psi, m_ks = {}, {}, {}, {}
    prog = st.progress(0, text="Simulating production months…")
    for mo in months:
        prod = drift_data(base_df, mo)
        metrics, _ = score_df(clf, prod, feats, thr)
        monthly[mo]   = prod
        m_metrics[mo] = metrics
        m_psi[mo]     = all_psi(base_df, prod, feats)
        m_ks[mo]      = {f: ks_test(base_df[f], prod[f]) for f in ["V1", "V2", "V3", "Amount"]}
        prog.progress(mo / n_months, text=f"Simulated month {mo} of {n_months}…")
    prog.empty()
    st.session_state["pipeline_done"] = True
    last = months[-1]

    # ══ TAB 0 — OVERVIEW ══
    with tabs[0]:
        st.subheader("Baseline Model Performance")
        cols = st.columns(5)
        for c, (k, v) in zip(cols, base_m.items()):
            c.metric(k, v)

        ca, cb = st.columns(2)
        with ca:
            fi = pd.DataFrame({"Feature": feats, "Gain": clf.feature_importances_}).nlargest(15, "Gain")
            fig = px.bar(fi, x="Gain", y="Feature", orientation="h", color="Gain",
                        color_continuous_scale="Blues", title="Top 15 Features by Gain")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=430, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            cd = base_df["Class"].value_counts()
            fig = px.pie(values=cd.values, names=["Legitimate", "Fraud"],
                        color_discrete_sequence=["#1565c0", "#c62828"], hole=0.35,
                        title=f"Class Distribution — Fraud Rate: {fr_pct*100:.2f}%")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Transaction Amount Distribution by Class")
        samp = base_df.sample(min(3000, len(base_df)), random_state=1)
        fig = px.histogram(samp, x="Amount", color="Class", nbins=60, barmode="overlay", opacity=0.7,
                           color_discrete_sequence=["#1565c0", "#c62828"], labels={"Class": "0=Legit / 1=Fraud"})
        st.plotly_chart(fig, use_container_width=True)

    # ══ TAB 1 — PERFORMANCE ══
    with tabs[1]:
        st.subheader("Model Performance Degradation Over Time")
        x_labels = ["Baseline"] + [f"M{m}" for m in months]
        keys, colors = ["AUC-ROC", "AUC-PR", "F1", "Recall"], ["#1565c0", "#2e7d32", "#e65100", "#b71c1c"]

        fig = make_subplots(rows=2, cols=2, subplot_titles=keys)
        for i, (k, col) in enumerate(zip(keys, colors)):
            r, c = divmod(i, 2)
            ys = [base_m.get(k)] + [m_metrics[m].get(k) for m in months]
            pairs = [(x, y) for x, y in zip(x_labels, ys) if y is not None]
            if pairs:
                vx, vy = zip(*pairs)
                fig.add_trace(go.Scatter(x=list(vx), y=list(vy), mode="lines+markers",
                             line=dict(color=col, width=2.5), marker=dict(size=9)), row=r+1, col=c+1)
            if k == "AUC-ROC":
                fig.add_hline(y=round(base_m["AUC-ROC"] - 0.05, 4), line_dash="dash",
                             line_color="red", annotation_text="Alert (-5pp)", row=1, col=1)
        fig.update_layout(height=500, showlegend=False, title_text="KPIs — Baseline → Production")
        st.plotly_chart(fig, use_container_width=True)

        rows = [{"Period": "Baseline", **base_m, "Fraud_%": round(fr_pct * 100, 3)}]
        rows += [{"Period": f"M{m}", **m_metrics[m]} for m in months]
        st.dataframe(pd.DataFrame(rows)[["Period", "AUC-ROC", "AUC-PR", "F1", "Prec", "Recall", "Fraud_%"]],
                    use_container_width=True)

        st.markdown("---")
        st.subheader("💰 Business Impact  *(100K daily transactions)*")
        r_drop = base_m["Recall"] - m_metrics[last]["Recall"]
        avg_a  = base_df[base_df["Class"] == 1]["Amount"].mean()
        missed = max(0, r_drop) * 100_000 * fr_pct
        c1, c2, c3 = st.columns(3)
        c1.metric("Recall Drop", f"{r_drop:+.2%}", delta_color="inverse")
        c2.metric("Missed Fraud / Day", f"{missed:,.0f}")
        c3.metric("Estimated Daily Exposure", f"${missed * avg_a:,.0f}")

    # ══ TAB 2 — PSI ══
    with tabs[2]:
        st.subheader("Population Stability Index (PSI)")
        st.caption("< 0.10 Stable · 0.10-0.20 Investigate · > 0.20 Model validity at risk")

        psi_mat = pd.DataFrame(m_psi).T
        psi_mat.index = [f"Month {m}" for m in psi_mat.index]
        top = psi_mat.mean().nlargest(15).index
        fig = px.imshow(psi_mat[top], zmin=0, zmax=0.35, text_auto=".3f", aspect="auto",
                        color_continuous_scale=[[0, "#1b5e20"], [0.10, "#66bb6a"], [0.17, "#ffd600"],
                                                [0.22, "#e53935"], [1, "#4a0000"]],
                        title="PSI Heatmap — Top 15 Features")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

        sel = st.selectbox("Select feature", feats)
        trend = [m_psi[m].get(sel, 0) for m in months]
        cvals = ["#1b5e20" if v < 0.10 else "#ffd600" if v < 0.20 else "#e53935" for v in trend]
        fig2 = go.Figure(go.Bar(x=[f"M{m}" for m in months], y=trend, marker_color=cvals,
                                text=[f"{v:.4f}" for v in trend], textposition="outside"))
        fig2.add_hline(y=0.10, line_dash="dot", line_color="orange")
        fig2.add_hline(y=0.20, line_dash="dot", line_color="red")
        fig2.update_layout(title=f"PSI Over Time — {sel}", height=320)
        st.plotly_chart(fig2, use_container_width=True)

        final_psi = m_psi[last]
        st.markdown(f"#### PSI Summary — Month {last}")
        st.dataframe(pd.DataFrame(
            [{"Feature": k, "PSI": v, "Status": psi_label(v)}
             for k, v in sorted(final_psi.items(), key=lambda x: -x[1])]
        ), use_container_width=True)

    # ══ TAB 3 — DRIFT ══
    with tabs[3]:
        st.subheader("Statistical Drift Detection — Kolmogorov-Smirnov Test")
        st.caption("p < 0.05 → the distribution shift is statistically significant at 95% confidence")

        ks_feats = list(m_ks[1].keys())
        ks_mat = pd.DataFrame({f"M{m}": {f: m_ks[m][f][0] for f in ks_feats} for m in months}).T
        fig = px.imshow(ks_mat, color_continuous_scale="RdYlGn_r", text_auto=".3f", aspect="auto",
                        title="KS Statistic — Higher = Greater Drift")
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame([
            {"Feature": f, "KS Statistic": m_ks[last][f][0], "p-value": m_ks[last][f][1],
             "Drift Detected": "🚨 Yes" if m_ks[last][f][2] else "✅ No"} for f in ks_feats
        ]), use_container_width=True)

        st.markdown("#### Distribution Comparison: Baseline vs Final Month")
        last_prod = monthly[last]
        c1, c2 = st.columns(2)
        for i, feat in enumerate(["V1", "V2", "V3", "Amount"]):
            lo, hi = base_df[feat].quantile(0.01), base_df[feat].quantile(0.99)
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=base_df[feat].clip(lo, hi), name="Baseline",
                                       opacity=0.6, marker_color="#1565c0", nbinsx=50))
            fig.add_trace(go.Histogram(x=last_prod[feat].clip(lo, hi), name=f"Month {last}",
                                       opacity=0.6, marker_color="#c62828", nbinsx=50))
            ks = m_ks[last][feat]
            fig.update_layout(barmode="overlay", height=270, legend=dict(orientation="h", y=1.15),
                              title=f"{feat}  |  KS={ks[0]:.3f}  p={ks[1]:.4f}")
            (c1 if i % 2 == 0 else c2).plotly_chart(fig, use_container_width=True)

    # ══ TAB 4 — ALERTS ══
    with tabs[4]:
        st.subheader("⚠️ Model Alert Center")
        alerts = gen_alerts(base_m, m_metrics[last], m_psi[last], monthly[last], fr_pct)
        bg = {"🔴 CRITICAL": "#ffcdd2", "🟡 WARNING": "#fff9c4", "🟢 CLEAR": "#c8e6c9"}
        border = {"🔴 CRITICAL": "#c62828", "🟡 WARNING": "#f57f17", "🟢 CLEAR": "#2e7d32"}

        for sev, cat, desc, action in alerts:
            st.markdown(f"""<div style="background:{bg.get(sev,'#f5f5f5')};padding:14px;border-radius:8px;
                margin:8px 0;border-left:5px solid {border.get(sev,'#999')}">
                <strong>{sev}</strong> &nbsp;|&nbsp; <em>{cat}</em><br>📌 {desc}<br>
                🔧 <strong>Recommended Action:</strong> {action}</div>""", unsafe_allow_html=True)

        crits = sum("🔴" in a[0] for a in alerts)
        warns = sum("🟡" in a[0] for a in alerts)
        status = "🔴 Action Required" if crits else "🟡 Monitor Closely" if warns else "🟢 Stable"
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Critical Alerts", crits, delta_color="inverse")
        c2.metric("Warnings", warns, delta_color="inverse")
        c3.metric("Overall Status", status)

    # ══ TAB 5 — MRM REPORT ══
    with tabs[5]:
        st.subheader("📋 Model Risk Management Report")
        st.caption("Auto-generated · SR 11-7 / RBI Model Risk Aligned · For Model Risk Committee")

        alerts     = gen_alerts(base_m, m_metrics[last], m_psi[last], monthly[last], fr_pct)
        crit_feats = [f for f, v in m_psi[last].items() if v >= 0.20]
        fin_auc    = m_metrics[last]["AUC-ROC"] or 0
        auc_delta  = round(base_m["AUC-ROC"] - fin_auc, 4)
        max_psi    = max(m_psi[last].values())
        live_fr    = monthly[last]["Class"].mean()
        fr_ratio   = live_fr / fr_pct
        rpt_date   = datetime.now().strftime("%d %B %Y")
        has_crit   = any("🔴" in a[0] for a in alerts)
        has_warn   = any("🟡" in a[0] for a in alerts)
        status_str = "⛔ Requires Immediate Action" if has_crit else "⚠️ Under Enhanced Monitoring" if has_warn else "✅ Valid for Continued Use"

        st.markdown(f"""
**Model ID:** FRAUD-LGB-001 &emsp;|&emsp; **Algorithm:** LightGBM &emsp;|&emsp; **Report Date:** {rpt_date}
**Validation Period:** {n_months} months &emsp;|&emsp; **Threshold:** {thr} &emsp;|&emsp; **Status:** {status_str}

---
### 1. Baseline Performance (Hold-out Validation)

| Metric | Value |
|---|---|
| AUC-ROC | {base_m["AUC-ROC"]} |
| AUC-PR | {base_m["AUC-PR"]} |
| F1 | {base_m["F1"]} |
| Precision | {base_m["Prec"]} |
| Recall | {base_m["Recall"]} |

### 2. Monitoring Findings — Month {last}

| Dimension | Finding |
|---|---|
| Population Stability | {len(crit_feats)} feature(s) with PSI ≥ 0.20: {', '.join(crit_feats) or 'None'} |
| Performance Trend | AUC-ROC {base_m["AUC-ROC"]} → {fin_auc} (Δ {auc_delta:+.4f}) |
| Fraud Rate | {live_fr*100:.3f}% vs baseline {fr_pct*100:.3f}% ({fr_ratio:.2f}× ratio) |

### 3. Risk Findings
""")
        for i, (sev, cat, desc, action) in enumerate(alerts, 1):
            st.markdown(f"**Finding {i}** [{sev}] *{cat}*  \n📌 {desc}  \n🔧 {action}\n")

        st.markdown(f"""
### 4. Conditions for Continued Use

| Condition | Threshold | Current | Status |
|---|---|---|---|
| Max feature PSI | < 0.20 | {max_psi:.4f} | {"🚨" if max_psi >= 0.20 else "✅"} |
| AUC-ROC decline | < 5pp | {auc_delta:+.4f} | {"🚨" if auc_delta >= 0.05 else "✅"} |
| Fraud rate ratio | < 2× | {fr_ratio:.2f}× | {"🚨" if fr_ratio > 2 else "✅"} |
| Recall degradation | < 10pp | {base_m["Recall"] - m_metrics[last]["Recall"]:+.4f} | {"🚨" if base_m["Recall"] - m_metrics[last]["Recall"] >= 0.10 else "✅"} |

### 5. Escalation Triggers

| Trigger | Threshold | Escalation |
|---|---|---|
| Feature PSI breach | ≥ 0.20 | Revalidation within 60 days |
| AUC-ROC decline | > 5pp | Escalate to Model Risk Committee |
| Fraud rate surge | > 2× baseline | Alert fraud ops + MRC |
| Recall degradation | > 10pp | Emergency model review |

---
*Generated by the MRMG Monitoring System · SR 11-7 / RBI Model Risk Aligned · Next validation due in {n_months} months.*
""")

        export = {
            "report_date": rpt_date, "model_id": "FRAUD-LGB-001", "monitoring_months": n_months,
            "baseline_metrics": base_m, "monthly_metrics": {str(k): v for k, v in m_metrics.items()},
            "final_psi": {k: float(v) for k, v in m_psi[last].items()},
            "alerts": [{"severity": a[0], "category": a[1], "description": a[2], "action": a[3]} for a in alerts],
        }
        st.download_button("📥 Download Full MRM Report (JSON)", data=json.dumps(export, indent=2, default=str),
                           file_name=f"mrm_report_{datetime.now():%Y%m%d}.json", mime="application/json")


if __name__ == "__main__":
    main()
