"""
CreditInsight v2: Consumer Default Risk Explorer
================================================
Explore a consumer-credit portfolio, compare simple risk models fairly, and
score a single account. Built with Python, pandas, scikit-learn, SQLite,
Streamlit and Plotly.

Data
  - Demo mode: SYNTHETIC data. Risk relationships are built into the generator,
    so demo results show that the pipeline works. They are NOT real findings.
  - Real mode: upload cs-training.csv from the Kaggle "Give Me Some Credit"
    competition (check the competition's terms of use).

Run:  pip install streamlit pandas numpy scikit-learn plotly
      streamlit run creditinsight_app.py
Optional: pip install groq   (only for the AI-written brief)

Author: Rutwik Satish
"""

import html
import io
import sqlite3
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CORE LOGIC (no UI calls below this line until the UI marker)
# ──────────────────────────────────────────────────────────────

RENAME = {
    "SeriousDlqin2yrs": "serious_delinquency",
    "RevolvingUtilizationOfUnsecuredLines": "revolving_util",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "pdue_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "pdue_90_plus",
    "NumberRealEstateLoansOrLines": "re_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "pdue_60_89",
    "NumberOfDependents": "dependents",
}
REQUIRED = [c for c in RENAME.values()]
PDUE_COLS = ["pdue_30_59", "pdue_60_89", "pdue_90_plus"]

BASE_FEATURES = [
    "revolving_util", "pdue_30_59", "debt_ratio", "monthly_income",
    "open_credit_lines", "pdue_90_plus", "re_loans", "pdue_60_89", "dependents",
]
ENGINEERED = [
    "util_high", "util_extreme", "any_90_late", "any_60_late", "any_30_late",
    "total_pdue_events", "high_debt_ratio", "income_low", "income_missing",
]
FEATURE_LABELS = {
    "revolving_util": "Revolving utilization", "pdue_30_59": "Count 30-59 days late",
    "debt_ratio": "Debt ratio", "monthly_income": "Monthly income",
    "open_credit_lines": "Open credit lines", "pdue_90_plus": "Count 90+ days late",
    "re_loans": "Real estate loans", "pdue_60_89": "Count 60-89 days late",
    "dependents": "Dependents", "age": "Age",
    "util_high": "Utilization > 75%", "util_extreme": "Utilization > 150%",
    "any_90_late": "Ever 90+ days late", "any_60_late": "Ever 60-89 days late",
    "any_30_late": "Ever 30-59 days late", "total_pdue_events": "Total past-due events",
    "high_debt_ratio": "Debt ratio > 1.0", "income_low": "Income < $3,000/mo",
    "income_missing": "Income not reported",
}
SEGMENT_ORDER = ["Low Risk", "Moderate", "High Risk", "Critical"]
SEGMENT_RULE = "Score 0-20 Low, 21-40 Moderate, 41-65 High, 66-100 Critical"


def generate_synthetic_data(n=150_000, seed=42):
    """SYNTHETIC data. Relationships are built in; results are not findings."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(52, 14, n).astype(int), 18, 90)
    pdue_30 = rng.poisson(0.42, n)
    pdue_60 = rng.poisson(0.24, n)
    pdue_90 = rng.poisson(0.21, n)
    low_mask = rng.random(n) < 0.72
    util = np.where(low_mask,
                    np.clip(rng.beta(0.8, 3, n), 0, 1),
                    np.clip(rng.beta(3, 1.2, n) * 1.5, 0, 3))
    debt_ratio = np.clip(rng.lognormal(-0.5, 1.1, n), 0, 3)
    income_raw = rng.lognormal(8.5, 0.6, n)
    income = np.where(rng.random(n) < 0.20, np.nan, income_raw)
    open_lines = np.clip(rng.poisson(8.5, n), 0, 40)
    re_loans = np.clip(rng.poisson(1.1, n), 0, 8)
    dependents = np.where(rng.random(n) < 0.025, np.nan, np.clip(rng.poisson(0.76, n), 0, 9))

    log_odds = (
        -4.0
        + 2.0 * (util > 0.75) + 1.0 * (util > 1.5)
        + 2.5 * (pdue_90 > 0) + 1.1 * (pdue_60 > 0) + 0.5 * (pdue_30 > 0)
        + 0.8 * (debt_ratio > 1.2) + 0.9 * (age < 30)
        - 0.7 * np.clip((age - 30) / 50, 0, 1)
        - 0.3 * np.clip(np.log1p(income_raw) / 11, 0, 1)
        + rng.normal(0, 0.25, n)
    )
    default = (rng.random(n) < 1 / (1 + np.exp(-log_odds))).astype(int)
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1), "serious_delinquency": default,
        "revolving_util": np.round(util, 4), "age": age, "pdue_30_59": pdue_30,
        "debt_ratio": np.round(debt_ratio, 4), "monthly_income": np.round(income, 0),
        "open_credit_lines": open_lines, "pdue_90_plus": pdue_90, "re_loans": re_loans,
        "pdue_60_89": pdue_60, "dependents": dependents,
    })


def clean_real_data(raw: pd.DataFrame):
    """Rename, validate and clean a Give Me Some Credit file. Returns (df, log)."""
    df = raw.rename(columns=RENAME).copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            "This file is missing expected columns: " + ", ".join(missing) +
            ". Use cs-training.csv from the Kaggle Give Me Some Credit competition.")
    log = [f"Loaded {len(df):,} rows."]
    # Rows with the label missing (e.g. the competition test file) cannot be used.
    n = len(df)
    df = df[df["serious_delinquency"].isin([0, 1])]
    if len(df) < n:
        log.append(f"Dropped {n - len(df):,} rows with no 0/1 default label.")
    n = len(df)
    df = df[(df["age"] >= 18) & (df["age"] <= 100)]
    if len(df) < n:
        log.append(f"Dropped {n - len(df):,} rows with age outside 18-100.")
    # Counts of 96/98 are known placeholder codes in this dataset, not real counts.
    for c in PDUE_COLS:
        bad = df[c] >= 90
        if bad.sum():
            log.append(f"{c}: set {int(bad.sum()):,} placeholder values (>= 90) to 0.")
            df.loc[bad, c] = 0
    over_util = int((df["revolving_util"] > 3).sum())
    over_debt = int((df["debt_ratio"] > 5).sum())
    if over_util:
        log.append(f"revolving_util: {over_util:,} values above 3.0 capped at 3.0.")
    if over_debt:
        log.append(f"debt_ratio: {over_debt:,} values above 5.0 capped at 5.0.")
    if "customer_id" not in df.columns:
        df["customer_id"] = np.arange(1, len(df) + 1)
    log.append(f"{len(df):,} rows kept for analysis.")
    return df.reset_index(drop=True), log


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add flags, a hand-weighted risk score and segments. Missing values stay missing."""
    df = df.copy()
    df["age"] = df["age"].clip(18, 100)
    df["revolving_util"] = df["revolving_util"].clip(0, 3)
    df["debt_ratio"] = df["debt_ratio"].clip(0, 5)
    df["util_high"] = (df["revolving_util"] > 0.75).astype(int)
    df["util_extreme"] = (df["revolving_util"] > 1.5).astype(int)
    df["any_90_late"] = (df["pdue_90_plus"] > 0).astype(int)
    df["any_60_late"] = (df["pdue_60_89"] > 0).astype(int)
    df["any_30_late"] = (df["pdue_30_59"] > 0).astype(int)
    df["total_pdue_events"] = df["pdue_30_59"] + df["pdue_60_89"] + df["pdue_90_plus"]
    df["high_debt_ratio"] = (df["debt_ratio"] > 1.0).astype(int)
    df["income_missing"] = df["monthly_income"].isna().astype(int)
    df["income_low"] = (df["monthly_income"] < 3000).astype(int)  # NaN -> 0
    df["age_group"] = pd.cut(df["age"], bins=[17, 29, 39, 49, 59, 100],
                             labels=["Under 30", "30-39", "40-49", "50-59", "60+"])
    # Hand-weighted score (judgment-based, NOT fitted). Age is deliberately excluded.
    df["risk_score"] = np.clip(
        df["revolving_util"].clip(0, 1) * 22
        + df["any_90_late"] * 28 + df["any_60_late"] * 14 + df["any_30_late"] * 8
        + df["debt_ratio"].clip(0, 2) / 2 * 14 + df["income_low"] * 8,
        0, 100).round(1)
    df["risk_segment"] = pd.cut(df["risk_score"], bins=[-1, 20, 40, 65, 100],
                                labels=SEGMENT_ORDER)
    return df


def model_features(use_age: bool):
    return BASE_FEATURES + ENGINEERED + (["age"] if use_age else [])


def train_models(df: pd.DataFrame, use_age: bool):
    """Train LR and a shallow tree on a stratified split. Imputation is fit on train only."""
    feats = model_features(use_age)
    X, y = df[feats], df["serious_delinquency"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=1000, random_state=42))
    lr.fit(X_tr, y_tr)
    dt = make_pipeline(SimpleImputer(strategy="median"),
                       DecisionTreeClassifier(max_depth=6, min_samples_leaf=150, random_state=42,
                                              class_weight="balanced"))
    dt.fit(X_tr, y_tr)
    lr_prob = lr.predict_proba(X_te)[:, 1]
    dt_prob = dt.predict_proba(X_te)[:, 1]
    util_te = X_te["revolving_util"].to_numpy()
    coefs = pd.DataFrame({"feature": feats, "coef": lr[-1].coef_[0]})
    coefs["label"] = coefs["feature"].map(FEATURE_LABELS).fillna(coefs["feature"])
    fpr, tpr, _ = roc_curve(y_te, lr_prob)
    return {
        "y_test": y_te.to_numpy(), "lr_prob": lr_prob, "dt_prob": dt_prob, "util": util_te,
        "lr": lr, "features": feats, "coefs": coefs,
        "auc": {"Logistic Regression": roc_auc_score(y_te, lr_prob),
                "Decision Tree": roc_auc_score(y_te, dt_prob),
                "Utilization only": roc_auc_score(y_te, util_te)},
        "roc": (fpr, tpr), "n_test": len(y_te), "base_rate": float(y_te.mean()),
    }


def metrics_at_rate(y, score, rate):
    """Flag the top `rate` share of accounts by score and report honest metrics."""
    thresh = np.quantile(score, 1 - rate)
    pred = (score >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "Flagged accounts": int(tp + fp), "Precision %": round(prec * 100, 1),
        "Recall %": round(rec * 100, 1),
        "False positive rate %": round(fp / (fp + tn) * 100, 1) if (fp + tn) else 0.0,
        "False alarms per true catch": round(fp / tp, 1) if tp else float("nan"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def segment_summary(conn):
    return pd.read_sql("""
        SELECT risk_segment, COUNT(*) AS accounts,
               ROUND(AVG(serious_delinquency) * 100, 1) AS default_rate_pct,
               ROUND(AVG(revolving_util) * 100, 1) AS avg_util_pct,
               ROUND(AVG(debt_ratio), 2) AS avg_debt_ratio,
               ROUND(AVG(age), 1) AS avg_age,
               SUM(serious_delinquency) AS total_defaults
        FROM customers GROUP BY risk_segment
        ORDER BY CASE risk_segment WHEN 'Low Risk' THEN 1 WHEN 'Moderate' THEN 2
                 WHEN 'High Risk' THEN 3 ELSE 4 END""", conn)


def build_report(ctx: dict) -> str:
    """Plain-text findings that only use computed numbers."""
    lines = [
        "# CreditInsight summary",
        f"Data: {ctx['source']}",
        f"Accounts analyzed: {ctx['n']:,}",
        f"Default definition: 90+ days past due within two years",
        f"Observed default rate: {ctx['default_rate']}%",
        f"High Risk + Critical segments: {ctx['hc_accounts_pct']}% of accounts, "
        f"{ctx['hc_default_share']}% of defaults",
        f"Hand-weighted risk score AUC: {ctx['score_auc']:.3f} (weights are judgment-based)",
        f"Models compared at a {ctx['rate_pct']}% flag rate on a held-out 25% test set:",
    ]
    for name, m in ctx["table"].items():
        lines.append(f"  - {name}: precision {m['Precision %']}%, recall {m['Recall %']}%, "
                     f"false positive rate {m['False positive rate %']}%")
    lines += [
        "Limits: single dataset, no time-based validation, no fairness testing, "
        "probabilities not calibrated for real lending decisions.",
    ]
    if ctx["synthetic"]:
        lines.append("WARNING: synthetic demo data. These results are not real-world findings.")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# ===== UI =====
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="CreditInsight · Default Risk Explorer", page_icon="💳",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:1.4rem;padding-bottom:2rem;}
[data-testid="metric-container"]{border:1px solid rgba(128,128,128,.3);border-radius:10px;padding:1rem 1.2rem;}
</style>""", unsafe_allow_html=True)

BLUE, GREEN, RED, AMBER = "#3B82F6", "#10B981", "#EF4444", "#F59E0B"
SEG_COLORS = {"Low Risk": GREEN, "Moderate": BLUE, "High Risk": AMBER, "Critical": RED}
LAYOUT = dict(margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(0,0,0,0)")


def show(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def table(df_):
    try:
        st.dataframe(df_, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df_, use_container_width=True, hide_index=True)


def label(text):
    st.markdown(f"**{text}**")


@st.cache_data(show_spinner=False)
def prepare(file_bytes):
    if file_bytes is None:
        return engineer_features(generate_synthetic_data()), ["Synthetic demo data generated."], True
    df, log = clean_real_data(pd.read_csv(io.BytesIO(file_bytes)))
    return engineer_features(df), log, False


@st.cache_resource(show_spinner=False)
def get_conn(_df, key):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    out = _df.copy()
    out["risk_segment"] = out["risk_segment"].astype(str)
    out["age_group"] = out["age_group"].astype(str)
    out.to_sql("customers", conn, index=False, if_exists="replace")
    return conn


@st.cache_resource(show_spinner=False)
def get_models(_df, key, use_age):
    return train_models(_df, use_age)


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💳 CreditInsight")
    st.caption("Consumer default risk explorer")
    mode = st.radio("Data", ["Demo (synthetic)", "Upload Give Me Some Credit CSV"])
    uploaded = None
    if mode.startswith("Upload"):
        uploaded = st.file_uploader("cs-training.csv", type=["csv"])
        st.caption("Get the file from the Kaggle competition page and check its terms.")
    use_age = st.checkbox("Use age as a model feature", value=False,
                          help="Off by default: age is sensitive in credit decisions and "
                               "regulated lenders restrict how it can be used. Age is still "
                               "shown in descriptive charts.")
    rate_pct = st.slider("Review capacity: share of accounts you can flag", 5, 50, 30, 5,
                         help="All models are compared at the same flag rate so the "
                              "comparison is fair.")
    groq_key = st.text_input("Groq API key (optional)", type="password",
                             help="Only used to write a short narrative brief.")

try:
    df, clean_log, is_synth = prepare(uploaded.getvalue() if uploaded is not None else None)
except ValueError as e:
    st.error(str(e))
    st.stop()
if mode.startswith("Upload") and uploaded is None:
    st.info("Upload a file in the sidebar, or switch to the demo data.")
    st.stop()

data_key = f"{is_synth}-{len(df)}-{int(df['serious_delinquency'].sum())}"
conn = get_conn(df, data_key)
seg = segment_summary(conn)
models = get_models(df, data_key, use_age)

if is_synth:
    st.warning("Demo mode: SYNTHETIC data. The generator builds risk relationships in, so these "
               "results show the pipeline works, not what is true of real borrowers. Upload the "
               "real file for actual results.", icon="⚠️")

# ── Shared numbers ────────────────────────────────────────────
n_accounts = len(df)
default_rate = round(df["serious_delinquency"].mean() * 100, 1)
avg_util = round(df["revolving_util"].mean() * 100, 1)
hc = df["risk_segment"].isin(["High Risk", "Critical"])
hc_accounts_pct = round(hc.mean() * 100, 1)
tot_def = df["serious_delinquency"].sum()
hc_default_share = round(df.loc[hc, "serious_delinquency"].sum() / max(tot_def, 1) * 100, 1)
seg["default_share"] = (seg["total_defaults"] / max(tot_def, 1) * 100).round(1)
score_auc = roc_auc_score(df["serious_delinquency"], df["risk_score"])
age_rates = df.groupby("age_group", observed=True)["serious_delinquency"].mean()
age_mult = (round(age_rates.get("Under 30", np.nan) / age_rates.get("40-49", np.nan), 1)
            if age_rates.get("40-49", 0) > 0 else None)

rate = rate_pct / 100
y_te = models["y_test"]
cmp_table = {
    "Logistic Regression": metrics_at_rate(y_te, models["lr_prob"], rate),
    "Decision Tree": metrics_at_rate(y_te, models["dt_prob"], rate),
    "Utilization only": metrics_at_rate(y_te, models["util"], rate),
}

tabs = st.tabs(["Overview", "Risk Segments", "Model Comparison", "Score an Account",
                "Briefing & Export"])

# ── Overview ──────────────────────────────────────────────────
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Accounts", f"{n_accounts:,}")
    c[1].metric("Default rate", f"{default_rate}%", help="90+ days past due within 2 years")
    c[2].metric("Avg revolving utilization", f"{avg_util}%")
    c[3].metric("High + Critical accounts", f"{hc_accounts_pct}%")

    left, right = st.columns(2)
    with left:
        label("Default vs non-default")
        vc = df["serious_delinquency"].map({0: "Non-default", 1: "Default"}).value_counts()
        fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.55,
                               marker_colors=[GREEN if i == "Non-default" else RED for i in vc.index]))
        fig.update_layout(**LAYOUT, height=240, showlegend=False)
        show(fig, "ov_pie")
    with right:
        label("Default rate by age group (descriptive only)")
        a = (age_rates * 100).round(1)
        fig = go.Figure(go.Bar(x=a.index.astype(str), y=a.values, marker_color=BLUE,
                               text=[f"{v}%" for v in a.values], textposition="outside"))
        fig.update_layout(**LAYOUT, height=240, yaxis=dict(showticklabels=False, showgrid=False))
        show(fig, "ov_age")
        st.caption("Age patterns describe the portfolio; they should not drive credit decisions.")

    label("Risk score distribution")
    fig = go.Figure()
    for name, val, col in [("Non-default", 0, GREEN), ("Default", 1, RED)]:
        fig.add_trace(go.Histogram(x=df.loc[df["serious_delinquency"] == val, "risk_score"],
                                   name=name, opacity=0.7, marker_color=col, nbinsx=40))
    fig.update_layout(**LAYOUT, height=220, barmode="overlay", xaxis_title="Risk score")
    show(fig, "ov_hist")

    with st.expander("Data preparation log"):
        for line in clean_log:
            st.write("• " + line)

# ── Risk Segments ─────────────────────────────────────────────
with tabs[1]:
    st.caption(f"Segments from a hand-weighted score (not fitted to data). {SEGMENT_RULE}. "
               "Age is not part of the score.")
    cols = st.columns(4)
    for col, (_, r) in zip(cols, seg.iterrows()):
        color = SEG_COLORS.get(r["risk_segment"], BLUE)
        col.markdown(
            f'<div style="border:1px solid rgba(128,128,128,.3);border-top:3px solid {color};'
            f'border-radius:10px;padding:12px;text-align:center;">'
            f'<b>{html.escape(str(r["risk_segment"]))}</b>'
            f'<div style="font-size:26px;font-weight:800;color:{color};">{r["default_rate_pct"]}%</div>'
            f'default rate<br>{int(r["accounts"]):,} accounts</div>', unsafe_allow_html=True)
    st.write("")
    l, r_ = st.columns(2)
    with l:
        label("Share of all defaults by segment")
        fig = go.Figure(go.Bar(x=seg["risk_segment"], y=seg["default_share"],
                               marker_color=[SEG_COLORS[s] for s in seg["risk_segment"]],
                               text=[f"{v}%" for v in seg["default_share"]], textposition="outside"))
        fig.update_layout(**LAYOUT, height=230, yaxis=dict(showticklabels=False, showgrid=False))
        show(fig, "sg_share")
    with r_:
        label("Average utilization by segment")
        fig = go.Figure(go.Bar(x=seg["risk_segment"], y=seg["avg_util_pct"],
                               marker_color=[SEG_COLORS[s] for s in seg["risk_segment"]],
                               text=[f"{v}%" for v in seg["avg_util_pct"]], textposition="outside"))
        fig.update_layout(**LAYOUT, height=230, yaxis=dict(showticklabels=False, showgrid=False))
        show(fig, "sg_util")
    out = seg[["risk_segment", "accounts", "default_rate_pct", "avg_util_pct", "avg_debt_ratio",
               "default_share"]].copy()
    out.columns = ["Segment", "Accounts", "Default rate %", "Avg util %", "Avg debt ratio",
                   "% of all defaults"]
    table(out)
    st.info(f"High Risk + Critical segments hold **{hc_accounts_pct}%** of accounts and "
            f"**{hc_default_share}%** of defaults.", icon="💡")
    st.caption(f"Does the hand-weighted score rank defaults? AUC = {score_auc:.3f} "
               "(0.5 = random, 1.0 = perfect).")

# ── Model Comparison ──────────────────────────────────────────
with tabs[2]:
    st.markdown(f"All models flag the **same top {rate_pct}%** of held-out test accounts "
                f"({models['n_test']:,} accounts, {models['base_rate']*100:.1f}% defaulted). "
                "Change the flag rate in the sidebar to match your review capacity.")
    rows = []
    for name, m in cmp_table.items():
        rows.append({"Model": name, **{k: m[k] for k in
                     ["Precision %", "Recall %", "False positive rate %",
                      "False alarms per true catch"]},
                     "AUC (all thresholds)": round(models["auc"][name], 3)})
    table(pd.DataFrame(rows))
    lr_m, base_m = cmp_table["Logistic Regression"], cmp_table["Utilization only"]
    st.markdown(
        f"At a {rate_pct}% flag rate, logistic regression catches **{lr_m['Recall %']}%** of "
        f"defaults versus **{base_m['Recall %']}%** for utilization alone, with "
        f"**{lr_m['False alarms per true catch']}** vs **{base_m['False alarms per true catch']}** "
        "false alarms per true catch.")
    l, r_ = st.columns(2)
    with l:
        label("What the logistic model leans on (standardized coefficients)")
        cf = models["coefs"].reindex(models["coefs"]["coef"].abs().sort_values().index).tail(10)
        fig = go.Figure(go.Bar(x=cf["coef"], y=cf["label"], orientation="h",
                               marker_color=[RED if v > 0 else GREEN for v in cf["coef"]]))
        fig.update_layout(**LAYOUT, height=320)
        show(fig, "md_coef")
        st.caption("Red raises predicted risk, green lowers it. Correlated features share credit, "
                   "so read this as direction, not exact importance.")
    with r_:
        label("ROC curve: logistic regression")
        fpr, tpr = models["roc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Logistic regression",
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="gray", dash="dash")))
        fig.update_layout(**LAYOUT, height=320, xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        show(fig, "md_roc")
    label(f"Logistic regression at the {rate_pct}% flag rate")
    table(pd.DataFrame({"": ["Actually defaulted", "Did not default"],
                        "Flagged": [lr_m["tp"], lr_m["fp"]],
                        "Not flagged": [lr_m["fn"], lr_m["tn"]]}))
    if is_synth:
        st.caption("Synthetic data: these metrics show the code works, not real model quality.")

# ── Score an Account ──────────────────────────────────────────
with tabs[3]:
    st.markdown("Enter an account profile to see the model's estimate. **Illustration only. "
                "Not a credit decision tool.**")
    a, b, c3 = st.columns(3)
    util_in = a.number_input("Revolving utilization (1.0 = 100%)", 0.0, 3.0, 0.35, 0.05)
    debt_in = a.number_input("Debt ratio", 0.0, 5.0, 0.4, 0.05)
    inc_in = a.number_input("Monthly income ($)", 0, 100000, 5000, 250)
    p30 = b.number_input("Times 30-59 days late", 0, 20, 0)
    p60 = b.number_input("Times 60-89 days late", 0, 20, 0)
    p90 = b.number_input("Times 90+ days late", 0, 20, 0)
    lines_in = c3.number_input("Open credit lines", 0, 40, 6)
    re_in = c3.number_input("Real estate loans", 0, 8, 1)
    dep_in = c3.number_input("Dependents", 0, 9, 1)
    age_in = c3.number_input("Age", 18, 100, 40)
    row = pd.DataFrame([{
        "customer_id": 0, "serious_delinquency": 0, "revolving_util": util_in, "age": age_in,
        "pdue_30_59": p30, "debt_ratio": debt_in, "monthly_income": float(inc_in),
        "open_credit_lines": lines_in, "pdue_90_plus": p90, "re_loans": re_in,
        "pdue_60_89": p60, "dependents": dep_in}])
    row = engineer_features(row)
    prob = float(models["lr"].predict_proba(row[models["features"]])[0, 1])
    pct_rank = float((models["lr_prob"] < prob).mean() * 100)
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated default probability", f"{prob*100:.1f}%")
    m2.metric("Riskier than", f"{pct_rank:.0f}% of test accounts")
    m3.metric("Hand-weighted segment", str(row["risk_segment"].iloc[0]),
              help=f"Score {row['risk_score'].iloc[0]}")
    coef = models["coefs"].set_index("feature")["coef"]
    scaler = models["lr"][1]
    imp = models["lr"][0]
    z = scaler.transform(imp.transform(row[models["features"]]))[0]
    contrib = pd.Series(z * coef.reindex(models["features"]).to_numpy(), index=models["features"])
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(4)
    st.markdown("**Main factors in this estimate** (relative to the average account)")
    for f, v in top.items():
        st.write(("↑ raises risk: " if v > 0 else "↓ lowers risk: ") + FEATURE_LABELS.get(f, f))
    st.caption("Probabilities come from a simple model on "
               + ("synthetic" if is_synth else "one public") + " dataset and are not calibrated "
               "for real lending.")

# ── Briefing & Export ─────────────────────────────────────────
with tabs[4]:
    ctx = dict(source="synthetic demo data" if is_synth else "uploaded Give Me Some Credit file",
               n=n_accounts, default_rate=default_rate, hc_accounts_pct=hc_accounts_pct,
               hc_default_share=hc_default_share, score_auc=score_auc, rate_pct=rate_pct,
               table=cmp_table, synthetic=is_synth)
    report = build_report(ctx)
    st.markdown("**Findings (computed from the data above)**")
    st.code(report, language="markdown")
    st.download_button("⬇ Download summary (.md)", report, "creditinsight_summary.md", "text/markdown")

    export = df[["customer_id", "age", "age_group", "revolving_util", "debt_ratio", "monthly_income",
                 "pdue_30_59", "pdue_60_89", "pdue_90_plus", "risk_score", "risk_segment",
                 "serious_delinquency"]]
    st.download_button("⬇ Download scored portfolio (.csv)", export.to_csv(index=False),
                       "creditinsight_scored_portfolio.csv", "text/csv")

    st.markdown("---")
    label("Optional: AI-written narrative")
    if not groq_key:
        st.caption("Add a Groq API key in the sidebar to draft a short narrative from the findings.")
    elif st.button("Draft narrative"):
        try:
            from groq import Groq
            prompt = (
                "You are a credit risk analyst. Using ONLY the facts below, write a plain-text "
                "brief of under 150 words: one summary sentence, two segments to prioritize, "
                "and one next step. Do not add numbers, causes or policies not stated. "
                "State that default means 90+ days past due within two years."
                + ("\nThe data is SYNTHETIC and must be described as a demo." if is_synth else "")
                + "\n\nFACTS:\n" + report)
            resp = Groq(api_key=groq_key).chat.completions.create(
                model="llama-3.3-70b-versatile", max_tokens=350,
                messages=[{"role": "user", "content": prompt}])
            text = html.escape(resp.choices[0].message.content.strip()).replace("\n", "<br>")
            st.markdown(f'<div style="border-left:3px solid {BLUE};padding:10px 14px;">{text}</div>',
                        unsafe_allow_html=True)
            st.caption("AI-drafted. Check every figure against the findings above before sharing.")
        except Exception as e:  # network, key or package problems
            st.error(f"Could not generate the narrative: {e}")

    with st.expander("Limits of this tool"):
        st.markdown(
            "- One dataset, one split. No time-based validation, so real-world drift is untested.\n"
            "- No fairness or disparate-impact testing. Do not use for real credit decisions.\n"
            "- Hand-weighted risk score is a judgment, not a fitted or validated scorecard.\n"
            "- Probabilities are uncalibrated and the demo data is synthetic.\n"
            "- Descriptive age patterns are not a basis for age-based policy.")
