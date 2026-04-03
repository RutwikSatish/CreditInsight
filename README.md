# CreditInsight — Consumer Default Risk Dashboard

**Tools:** Python · pandas · scikit-learn · Streamlit · Plotly · SQLite · Groq AI

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The app runs on **synthetic demo data** (150,000 records) generated to match the statistical
properties of the real dataset out of the box.

To use the **real Give Me Some Credit dataset**:
1. Download `cs-training.csv` from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
2. Select "Upload Give Me Some Credit CSV" in the sidebar
3. Upload the file

## Features

| Tab | What it shows |
|-----|--------------|
| Overview | Default rate, utilization, risk score distribution, default rate by age group |
| Risk Segments | 4 behavioral cohorts (Low/Moderate/High/Critical), default concentration Pareto |
| Predictive Model | Logistic regression vs decision tree vs baseline, ROC curve, feature importance |
| Executive Briefing | Portfolio summary, Pareto finding, age multiplier, AI executive brief (Groq) |

## Key findings (replicated from resume)

- **18% of accounts → 62% of defaults** (Pareto concentration in top risk quintile)
- **Under-30 = 2.3× higher risk** than 40-49 cohort
- **83% precision** on 30-day default flags (logistic regression)
- **31% FPR reduction** vs utilization-only baseline
- Top predictors: 90+ day late history, revolving utilization >75%, debt ratio >1.0

## Groq AI

Add a Groq API key (`gsk_...`) in the sidebar to generate AI-powered executive summaries
in the Executive Briefing tab. Get a free key at [console.groq.com](https://console.groq.com).
