"""Look and feel: CSS + small HTML blocks injected into the Streamlit page."""
import matplotlib
import streamlit as st
from cycler import cycler

matplotlib.rcParams.update({
    "axes.prop_cycle": cycler(color=["#4f46e5", "#0d9488", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#84cc16", "#ec4899"]),
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.facecolor": "#fcfcfe",
    "axes.titleweight": "bold", "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "legend.frameon": False})

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"], button, input, textarea, select { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
.stApp { background: radial-gradient(1100px 420px at 8% -8%, #e0e7ff 0%, transparent 60%), radial-gradient(900px 380px at 100% 0%, #ccfbf1 0%, transparent 55%), #f6f7fb; }
.block-container { max-width: 1180px; padding-top: 1.1rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
footer, #MainMenu, [data-testid="stAppDeployButton"], .stDeployButton { visibility: hidden; display: none; }


/* readable text even if the browser/Streamlit is in dark mode */
.stApp { color: #1f2937; color-scheme: light; }
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #374151 !important; }
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] li p { color: #1f2937; }
.block-container h1, .block-container h2, .block-container h3, .block-container h4 { color: #312e81; }
.st-key-nav div[role="radiogroup"] > label p { color: #374151; }

/* hero */
.hero { position: relative; overflow: hidden; color: #fff; padding: 30px 34px; border-radius: 22px; margin-bottom: 18px;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 48%, #0d9488 100%); box-shadow: 0 18px 40px rgba(79,70,229,.28); }
.hero::before { content: ""; position: absolute; right: -70px; top: -80px; width: 260px; height: 260px; border-radius: 50%; background: rgba(255,255,255,.12); }
.hero::after { content: ""; position: absolute; right: 120px; bottom: -110px; width: 220px; height: 220px; border-radius: 50%; background: rgba(255,255,255,.09); }
.hero-badge { display: inline-block; font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; padding: 4px 12px; border-radius: 999px; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); }
.hero-title { font-size: 2.35rem; font-weight: 800; line-height: 1.15; margin: 0 0 6px; color: #fff; }
.hero-sub { font-size: 1.02rem; max-width: 640px; opacity: .93; margin: 0 0 14px; color: #fff; }
.chips { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 8px; }
.chips span { font-size: .78rem; font-weight: 500; padding: 5px 12px; border-radius: 999px; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3); }

/* step cards (landing page) */
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 14px 0 6px; }
.step { background: #fff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px 18px; box-shadow: 0 6px 18px rgba(15,23,42,.06); }
.step b { display: inline-grid; place-items: center; width: 28px; height: 28px; border-radius: 50%; color: #fff; font-size: .85rem; margin-right: 8px; background: linear-gradient(135deg, #4f46e5, #7c3aed); }
.step h4 { display: inline; font-size: 1.02rem; color: #312e81; }
.step p { margin: 8px 0 0; color: #4b5563; font-size: .9rem; }

/* section navigation pills */
.st-key-nav div[role="radiogroup"] { gap: .45rem; flex-wrap: wrap; background: #fff; padding: .55rem; border-radius: 18px; border: 1px solid #e5e7eb; box-shadow: 0 6px 18px rgba(15,23,42,.06); }
.st-key-nav div[role="radiogroup"] > label { margin: 0; padding: .38rem 1rem; border-radius: 999px; background: #f3f4f6; border: 1px solid transparent; cursor: pointer; transition: all .15s ease; }
.st-key-nav div[role="radiogroup"] > label > div:first-child { display: none; }
.st-key-nav div[role="radiogroup"] > label:hover { background: #eef2ff; border-color: #c7d2fe; }
.st-key-nav div[role="radiogroup"] > label:has(input:checked) { background: linear-gradient(135deg, #4f46e5, #7c3aed); box-shadow: 0 8px 18px rgba(79,70,229,.35); }
.st-key-nav div[role="radiogroup"] > label:has(input:checked) * { color: #fff !important; font-weight: 600; }

/* section banner */
.sec { display: flex; align-items: center; gap: 14px; background: #fff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 12px 16px; margin: 14px 0 16px; box-shadow: 0 6px 18px rgba(15,23,42,.05); }
.sec-ico { flex: none; width: 48px; height: 48px; display: grid; place-items: center; font-size: 1.5rem; border-radius: 14px; background: linear-gradient(135deg, #eef2ff, #ccfbf1); }
.sec-t { font-weight: 700; color: #312e81; font-size: 1.08rem; }
.sec-d { color: #4b5563; font-size: .9rem; }

/* widgets */
[data-testid="stMetric"] { background: #fff; border: 1px solid #e5e7eb; border-left: 5px solid #4f46e5; border-radius: 14px; padding: .75rem 1rem; box-shadow: 0 6px 16px rgba(15,23,42,.05); }
[data-testid="stMetricLabel"] p { color: #6b7280; font-weight: 500; }
[data-testid="stMetricValue"] { color: #312e81; font-weight: 700; }
.stButton > button { border: 0; border-radius: 12px; padding: .55rem 1.2rem; color: #fff; font-weight: 600; background: linear-gradient(135deg, #4f46e5, #7c3aed); box-shadow: 0 8px 18px rgba(79,70,229,.30); transition: transform .12s ease, box-shadow .12s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 22px rgba(79,70,229,.38); color: #fff; }
.stButton > button p, .stDownloadButton > button p { color: #fff; }
.stDownloadButton > button { border: 0; border-radius: 12px; padding: .55rem 1.2rem; color: #fff; font-weight: 600; width: 100%; background: linear-gradient(135deg, #0d9488, #14b8a6); box-shadow: 0 8px 18px rgba(13,148,136,.30); }
.stDownloadButton > button:hover { color: #fff; transform: translateY(-1px); }
[data-testid="stFileUploader"] section { border: 2px dashed #a5b4fc; background: #fff; border-radius: 16px; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 12px; overflow: hidden; border: 1px solid #e5e7eb; box-shadow: 0 4px 14px rgba(15,23,42,.05); background: #fff; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stImage"] img { border-radius: 12px; border: 1px solid #e5e7eb; background: #fff; padding: 6px; box-shadow: 0 4px 14px rgba(15,23,42,.05); }
.block-container h3 { color: #312e81; border-left: 5px solid #7c3aed; padding-left: .65rem; margin-top: 1.2rem; }
.foot { text-align: center; color: #6b7280; font-size: .82rem; margin-top: 2.2rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; }
@media (max-width: 640px) { .hero { padding: 22px 18px; } .hero-title { font-size: 1.7rem; } }
"""

HELP = {
    "Data": ("📋", "Data overview", "Preview of your file, column types and missing values."),
    "EDA": ("📊", "Exploratory analysis", "Histogram, boxplot and Q-Q plot, correlation heatmap, scatter and group plots, each with an interpretation."),
    "Test advisor": ("🧭", "Which test should I use?", "Checks normality and suggests parametric or non-parametric tests for your goal."),
    "PCA": ("🎯", "Principal component analysis", "Finds the main sources of variation and the variables that matter most."),
    "Regression": ("📈", "Linear regression", "Numeric and categorical predictors, fitted line, residual diagnostics and an optional hold-out test."),
    "Two-group test": ("⚖️", "Compare two groups", "Welch t-test with Mann-Whitney as a check, plus effect sizes."),
    "Paired / one-sample": ("🔁", "Paired and one-sample tests", "Paired t-test or Wilcoxon, one-sample t-test or Wilcoxon, and the Friedman test for 3+ related columns."),
    "ANOVA": ("🧪", "Compare three or more groups", "Welch ANOVA, Kruskal-Wallis and Holm-adjusted pairwise tests."),
    "Chi-square": ("🔗", "Association of categories", "Chi-square test, Cramer's V and standardised residuals."),
    "Report": ("📄", "Download a report", "Build one report and download it as Excel, PDF or Word."),
}


def inject():
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def hero():
    st.markdown("""<div class="hero">
<div class="hero-title">StatGuide</div>
<p class="hero-sub">Upload a data file and get statistics, PCA, regression and a complete report, with plain-language interpretation at every step.</p>
<div class="chips"><span>EDA &amp; Q-Q plots</span><span>Normality &amp; test advisor</span><span>PCA</span><span>Regression</span><span>t-test &middot; ANOVA &middot; Chi-square</span><span>Excel &middot; PDF &middot; Word report</span></div></div>""", unsafe_allow_html=True)


def steps():
    st.markdown("""<div class="steps">
<div class="step"><b>1</b><h4>Upload</h4><p>Choose the file type (CSV, Excel, text or JSON) and drop your file above.</p></div>
<div class="step"><b>2</b><h4>Explore</h4><p>Move through the sections: EDA, test advisor, PCA, regression and tests.</p></div>
<div class="step"><b>3</b><h4>Report</h4><p>Download everything as an Excel, PDF or Word report.</p></div></div>""", unsafe_allow_html=True)


def section_banner(name):
    ico, title, desc = HELP.get(name, ("", name, ""))
    st.markdown(f'<div class="sec"><div class="sec-ico">{ico}</div><div><div class="sec-t">{title}</div><div class="sec-d">{desc}</div></div></div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="foot">StatGuide &middot; interpretations are rules of thumb, so check the plots and the assumptions yourself. Do not upload personal or confidential data to a public copy of this app.</div>', unsafe_allow_html=True)
