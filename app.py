import io
import numpy as np
import pandas as pd
import streamlit as st

from stats_tools import (two_group_test, multi_group_test, interpret_anova, chi_square_test, interpret_chi,
                         paired_test, one_sample_test, friedman_test, interpret_paired, interpret_one_sample, interpret_friedman, fmt_p)
import eda_tools as E
from advisor_tools import GOALS, normality_table, group_normality, overall, recommend
import pca_tools as PC
import regression_tools as RG
import io_tools as IO
import report_tools as RP
import theme


@st.cache_data(show_spinner=False)
def load_table(raw, kind, sep_name, sheet):
    return IO.read_table(raw, kind, sep_name, sheet)


@st.cache_data(show_spinner=False)
def cached_diag_pngs(d, y, xs):
    """Residual-diagnostic plots drawn once per (data, model) and reused as small PNG images."""
    import io
    r = RG.fit_ols(d, y, list(xs)); out = []
    for f in RG.diag_figs(r):
        b = io.BytesIO(); f.savefig(b, format="png", dpi=110); out.append(b.getvalue()); RG.plt.close(f)
    return out


@st.cache_data(show_spinner=False)
def cached_norm(d, alpha):
    return normality_table(d, alpha)


@st.cache_data(show_spinner=False)
def cached_pca(d, cols, impute, log_skewed):
    return PC.run_pca(d, list(cols), impute, log_skewed)

st.set_page_config(page_title="StatGuide", layout="wide")
theme.inject()
theme.hero()

kind = st.selectbox("File type", IO.KINDS, key="ftype")
file = st.file_uploader("Upload your data file", type=IO.EXTS[kind], key="up")
if file is None:
    st.info("Choose the file type, then upload a CSV, text, Excel or JSON file. Try a small public dataset (e.g. Heart Failure from UCI).")
    theme.steps(); theme.footer()
    st.stop()
raw, fname = file.getvalue(), getattr(file, "name", "data")
try:
    eff = IO.detect_kind(fname) if kind == "Auto-detect" else kind
    sep_name, sheet = "Auto-detect", None
    if eff in ("CSV", "Text / TSV"):
        sep_name = st.selectbox("Delimiter", list(IO.SEPS), key="sep")
    if eff == "Excel":
        sheets = IO.excel_sheets(raw)
        sheet = st.selectbox("Excel sheet", sheets, key="sheet") if len(sheets) > 1 else sheets[0]
    df = load_table(raw, eff, sep_name, sheet)
except ValueError as e:
    st.error(str(e)); st.stop()
st.caption(f"Loaded {fname} as {eff}: {df.shape[0]} rows x {df.shape[1]} columns")
if df.shape[1] == 1 and eff in ("CSV", "Text / TSV"):
    st.warning("Only 1 column was found. The delimiter may be wrong: choose it manually in the Delimiter box above.")
NAMES = ["Data", "EDA", "Test advisor", "PCA", "Regression", "Two-group test", "Paired / one-sample", "ANOVA", "Chi-square", "Report"]
nav = st.radio("Section", NAMES, horizontal=True, key="nav", label_visibility="collapsed")
theme.section_banner(nav)

# ---------------------------------------------------------------- Data
if nav == "Data":
    st.dataframe(df.head(20))
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing cells", int(df.isna().sum().sum()))
    st.write("Column types and missing values")
    st.dataframe(E.profile_columns(df))
    st.caption("Columns marked 'excluded' are IDs or constants. Binary, categorical and missing values are shown for information; each analysis handles missing values on its own.")
    st.write("Summary statistics")
    st.dataframe(df.describe(include="all").T)

# ------------------------------------------------------------------ EDA
if nav == "EDA":
    num_cols, cat_cols = E.split_columns(df)
    st.subheader("1. Missing values")
    fm = E.missing_fig(df)
    if fm is None:
        st.success("No missing values.")
    else:
        st.pyplot(fm); st.caption("Columns with many missing values may need dropping or imputation.")
    if num_cols:
        st.subheader("2. Distribution of one numeric variable")
        col = st.selectbox("Numeric variable", num_cols, key="eda_num")
        if df[col].notna().sum() < 8:
            st.warning("Need at least 8 non-missing values.")
        else:
            f1, f2, f3 = E.dist_figures(df[col], col)
            c1, c2, c3 = st.columns(3); c1.pyplot(f1); c2.pyplot(f2); c3.pyplot(f3)
            st.caption("Q-Q plot: points on the line = normal-like; S-shape or curled ends = heavy or light tails; a bow = skewness.")
            st.markdown("**Interpretation**")
            for line in E.interpret_numeric(col, E.summarize(df[col])):
                st.markdown("- " + line)
    if len(num_cols) >= 2:
        st.subheader("3. Relationships between numeric variables")
        method = st.radio("Correlation type", ["pearson", "spearman"], horizontal=True, key="eda_m")
        sel = st.multiselect("Variables for the heatmap (2 to 20)", num_cols, default=num_cols[:12],
                             max_selections=20, key="eda_sel")
        if len(sel) >= 2:
            fig, cm = E.corr_heatmap(df, sel, method); st.pyplot(fig)
            tp = E.top_pairs(cm); st.dataframe(tp)
            for line in E.interpret_pairs(tp):
                st.markdown("- " + line)
        cx, cy = st.columns(2)
        xv = cx.selectbox("Scatter X", num_cols, key="eda_x")
        yv = cy.selectbox("Scatter Y", num_cols, index=1, key="eda_y")
        if xv != yv:
            fs, pr, sr = E.scatter_fig(df, xv, yv); st.pyplot(fs)
            for line in E.interpret_scatter(xv, yv, pr, sr):
                st.markdown("- " + line)
    if num_cols and cat_cols:
        st.subheader("4. Numeric variable by group")
        gn = st.selectbox("Numeric", num_cols, key="eda_gn"); gc = st.selectbox("Group", cat_cols, key="eda_gc")
        fg, tab = E.group_box(df, gn, gc); st.pyplot(fg); st.dataframe(tab)
        st.caption("Overlapping boxes are an informal check only. Use the Two-group test tab for a formal test.")
    if cat_cols:
        st.subheader("5. Categorical variable")
        cc = st.selectbox("Categorical", cat_cols, key="eda_cc")
        fc, msg = E.cat_bar(df[cc].dropna(), cc); st.pyplot(fc); st.markdown("- " + msg)
    st.caption("Skewness, kurtosis and correlation cut-offs are rules of thumb, not strict thresholds.")
    E.plt.close("all")

# ------------------------------------------------------------------ PCA
if nav == "PCA":
    nc, cc = E.split_columns(df)
    if len(nc) < 3:
        st.warning("PCA needs at least 3 numeric variables.")
    else:
        sel = st.multiselect("Numeric variables", nc, default=nc[:30], key="pca_sel")
        c1, c2 = st.columns(2)
        imp_mode = c1.radio("Missing values", ["Fill with median", "Drop incomplete rows"], key="pca_imp")
        grp = c2.selectbox("Colour the biplot by (optional)", ["(none)"] + [c for c in cc if df[c].nunique() <= 10], key="pca_grp")
        logs = st.checkbox("Apply log(1+x) to skewed non-negative variables (|skewness| > 1) before PCA", key="pca_log")
        mode = st.radio("How many components to keep?", ["Cumulative variance threshold", "Kaiser rule (eigenvalue > 1)", "I choose the number"],
                        horizontal=True, key="pca_mode")
        val = 80
        if mode.startswith("Cumulative"):
            val = st.slider("Keep components until this % of variance is explained", 50, 99, 80, key="pca_thr")
        elif mode.startswith("I choose"):
            val = st.number_input("Number of components", 1, max(1, len(sel)), min(2, max(1, len(sel))), key="pca_m")
        try:
            res = cached_pca(df, tuple(sel), imp_mode.startswith("Fill"), logs)
        except ValueError as e:
            st.error(str(e)); st.stop()
        m = PC.choose_m(res, mode, val); suit = PC.suitability(res); L = PC.loadings(res, m); imp = PC.importance(res, m)
        a1, a2, a3 = st.columns(3)
        a1.metric("Rows used", res["n"]); a2.metric("KMO", f"{suit['kmo']:.2f}"); a3.metric("Bartlett p", f"{suit['p']:.3g}")
        st.pyplot(PC.scree_fig(res, m))
        st.dataframe(pd.DataFrame({"component": [f"PC{i + 1}" for i in range(res["p"])], "eigenvalue": res["eig"].round(3),
                                   "% variance": (res["ratio"] * 100).round(2), "cumulative %": (res["cum"] * 100).round(2)}).head(15))
        st.subheader("Which variables are important?")
        st.pyplot(PC.importance_fig(imp)); st.dataframe(imp)
        st.pyplot(PC.loadings_fig(L)); st.dataframe(L.round(3))
        st.pyplot(PC.biplot(res, None if grp == "(none)" else df[grp]))
        st.markdown("**Interpretation**")
        for line in PC.interpret_pca(res, m, imp, suit):
            st.markdown("- " + line)
        st.caption("PCA is unsupervised and works on standardised variables: importance means contribution to the main sources of variation, "
                   "not predictive power for a target. It captures linear structure only and is sensitive to outliers. KMO and loading cut-offs are rules of thumb.")
        PC.plt.close("all")

# ------------------------------------------------------------ Regression
if nav == "Regression":
    nc, cc = E.split_columns(df)
    if len(nc) < 2:
        st.warning("Regression needs at least 2 numeric variables.")
    else:
        ry = st.selectbox("Response (Y)", nc, key="rg_y")
        opts = [c for c in nc if c != ry]
        rx = st.multiselect("Numeric predictor(s) (X): choose one numeric predictor to see the regression line", opts, default=opts[:1], key="rg_x")
        cat_opts = [c for c in cc if 2 <= df[c].nunique(dropna=True) <= 10 and c != ry]
        rcat = st.multiselect("Categorical predictor(s) (optional): dummy-coded, the first level is the reference", cat_opts, key="rg_cat")
        o1, o2 = st.columns(2)
        ordered = o1.checkbox("My rows are in time order (show Durbin-Watson)", key="rg_dw")
        holdout = o2.checkbox("Check accuracy on a hold-out test set", key="rg_ho")
        if holdout:
            h1, h2 = st.columns(2)
            frac = h1.slider("Share of rows kept for testing (%)", 10, 40, 25, key="rg_frac")
            seed = h2.number_input("Random seed", 0, 9999, 42, key="rg_seed")
        if rx or rcat:
            dfr, dums = RG.add_dummies(df, rcat) if rcat else (df, [])
            xs = list(rx) + dums
            try:
                r = RG.fit_ols(dfr, ry, xs)
            except ValueError as e:
                st.error(str(e)); st.stop()
            r["dummies"] = dums
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows used", r["n"]); m2.metric("R-squared", f"{r['r2']:.3f}"); m3.metric("Adj. R-squared", f"{r['adj_r2']:.3f}"); m4.metric("F-test p", RG.pf(r["pF"]))
            st.pyplot(RG.line_fig(r) if (r["k"] == 1 and not dums) else RG.avp_fig(r))
            st.dataframe(r["coef"].round(4))
            st.subheader("Residual diagnostics")
            g1, g2, g3 = st.columns(3)
            for g, png in zip((g1, g2, g3), cached_diag_pngs(dfr, ry, tuple(xs))):
                g.image(png, use_container_width=True)
            st.markdown("**Interpretation**")
            for line in RG.interpret(r, show_dw=ordered):
                st.markdown("- " + line)
            if holdout:
                st.subheader("Hold-out test")
                try:
                    ho = RG.holdout_eval(dfr, ry, xs, frac / 100, int(seed))
                    k1, k2, k3 = st.columns(3); k1.metric("Test RMSE", f"{ho['rmse_test']:.4g}"); k2.metric("Test R-squared", f"{ho['r2_test']:.3f}"); k3.metric("Training R-squared", f"{ho['r2_train']:.3f}")
                    for line in RG.interpret_holdout(ho):
                        st.markdown("- " + line)
                except ValueError as e:
                    st.warning(str(e))
            st.caption("Association is not causation, and predictions outside the observed X range are unreliable. Check the plots as well as the tests. "
                       "Rows with missing values in the chosen columns are dropped. Categorical predictors are dummy-coded (name=level); the diagnostics assume a linear model with independent errors.")
            RG.plt.close("all")

# ------------------------------------------------------- Two-group test
if nav == "Two-group test":
    num_cols = df.select_dtypes("number").columns.tolist()
    grp_cols = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    if not num_cols or not grp_cols:
        st.warning("Need one numeric column and one column with exactly 2 groups.")
    else:
        value = st.selectbox("Numeric column to compare", num_cols)
        group = st.selectbox("Grouping column (exactly 2 groups)", grp_cols)
        if st.button("Run two-group test"):
            levels = sorted(df[group].dropna().unique(), key=str)
            x = df.loc[df[group] == levels[0], value]
            y = df.loc[df[group] == levels[1], value]
            r = two_group_test(x, y)
            st.write(f"Group 1 = **{levels[0]}** (n={r['n1']}, mean={r['mean1']:.3f})   |   "
                     f"Group 2 = **{levels[1]}** (n={r['n2']}, mean={r['mean2']:.3f})")
            lo, hi = r["mean_diff_CI"]
            st.table(pd.DataFrame({
                "Result": ["Welch t-test p-value", "95% CI of mean difference (group1 - group2)",
                           "Cohen's d", "Mann-Whitney p-value (sensitivity check)",
                           "Rank-biserial correlation"],
                "Value": [fmt_p(r['welch_p']), f"({lo:.3f}, {hi:.3f})",
                          f"{r['cohen_d']:.3f}", fmt_p(r['mannwhitney_p']),
                          f"{r['rank_biserial']:.3f}"]}))
            st.markdown("**Assumption information (not a gate)**")
            st.write(f"Shapiro-Wilk p: group 1 = {r['shapiro_p_group1']:.3f}, "
                     f"group 2 = {r['shapiro_p_group2']:.3f}  |  Levene p = {r['levene_p']:.3f}")
            st.caption("Welch is the default because it does not assume equal variances. "
                       "Check both p-values: if Welch and Mann-Whitney disagree, look at the "
                       "distributions before deciding. Running many tests inflates false positives.")

# ------------------------------------------------------ Paired / one-sample
if nav == "Paired / one-sample":
    num_all = df.select_dtypes("number").columns.tolist()
    pmode = st.radio("What do you want to do?", ["Paired test (2 related columns)", "One-sample test (a column against a value)", "Friedman test (3+ related columns)"], key="pr_mode")
    if pmode.startswith("Paired"):
        if len(num_all) < 2:
            st.warning("Need at least 2 numeric columns (for example before and after).")
        else:
            pa = st.selectbox("First measurement (for example before)", num_all, key="pr_a"); pb = st.selectbox("Second measurement (for example after)", num_all, index=1, key="pr_b")
            if st.button("Run paired test"):
                if pa == pb:
                    st.error("Choose two different columns.")
                else:
                    try:
                        r = paired_test(df[pa].to_numpy(float), df[pb].to_numpy(float))
                        st.write(f"{r['n']} complete pairs | mean {pa} = {r['mean1']:.4g}, mean {pb} = {r['mean2']:.4g}")
                        st.table(pd.DataFrame({"Result": ["Paired t-test p-value", "Mean difference (first - second)", "95% CI of the difference", "Cohen's dz",
                                                          "Wilcoxon signed-rank p-value (sensitivity check)", "Matched-pairs rank-biserial", "Shapiro-Wilk p of the differences"],
                                               "Value": [fmt_p(r['p_t']), f"{r['mean_diff']:.4g}", f"({r['ci'][0]:.4g}, {r['ci'][1]:.4g})", f"{r['cohen_dz']:.3f}",
                                                         "-" if np.isnan(r["p_wilcoxon"]) else fmt_p(r['p_wilcoxon']), "-" if np.isnan(r["rank_biserial"]) else f"{r['rank_biserial']:.3f}",
                                                         "-" if np.isnan(r["shapiro_p_diff"]) else fmt_p(r['shapiro_p_diff'])]}))
                        xa, ya = df[pa].to_numpy(float), df[pb].to_numpy(float); ok = ~(np.isnan(xa) | np.isnan(ya))
                        fg, ax = E.plt.subplots(figsize=(4.5, 3)); ax.hist(xa[ok] - ya[ok], bins=20, alpha=.8); ax.axvline(0, color="red"); ax.set_title("Differences (first - second)"); fg.tight_layout()
                        st.pyplot(fg); E.plt.close("all")
                        st.markdown("**Interpretation**")
                        for line in interpret_paired(r, pa, pb):
                            st.markdown("- " + line)
                        st.caption("The paired t-test assumes the differences (not the raw columns) are roughly normal. Use it only when each row is the same subject measured twice.")
                    except ValueError as e:
                        st.error(str(e))
    elif pmode.startswith("One-sample"):
        if not num_all:
            st.warning("Need a numeric column.")
        else:
            oc = st.selectbox("Numeric column", num_all, key="os_c"); mu0 = st.number_input("Value to compare with (hypothesised mean)", value=0.0, key="os_mu")
            if st.button("Run one-sample test"):
                try:
                    r = one_sample_test(df[oc].to_numpy(float), float(mu0))
                    st.write(f"n = {r['n']} | mean = {r['mean']:.4g}, median = {r['median']:.4g}, sd = {r['sd']:.4g}")
                    st.table(pd.DataFrame({"Result": ["One-sample t-test p-value", "95% CI of the mean", "Cohen's d", "Wilcoxon signed-rank p-value (sensitivity check)", "Shapiro-Wilk p"],
                                           "Value": [fmt_p(r['p_t']), f"({r['ci'][0]:.4g}, {r['ci'][1]:.4g})", f"{r['cohen_d']:.3f}",
                                                     "-" if np.isnan(r["p_wilcoxon"]) else fmt_p(r['p_wilcoxon']), "-" if np.isnan(r["shapiro_p"]) else fmt_p(r['shapiro_p'])]}))
                    st.markdown("**Interpretation**")
                    for line in interpret_one_sample(r):
                        st.markdown("- " + line)
                except ValueError as e:
                    st.error(str(e))
    else:
        fcols = st.multiselect("Related measurement columns (3 or more)", num_all, default=num_all[:3], key="fr_cols")
        if st.button("Run Friedman test"):
            if len(fcols) < 3:
                st.error("Choose at least 3 columns.")
            else:
                try:
                    r = friedman_test(df[fcols])
                    st.write(f"{r['n']} complete subjects (rows), {r['k']} conditions")
                    st.dataframe(r["summary"]); st.markdown("**Pairwise Wilcoxon signed-rank tests, Holm-adjusted**"); st.dataframe(r["pairs"])
                    st.markdown("**Interpretation**")
                    for line in interpret_friedman(r):
                        st.markdown("- " + line)
                    st.caption("Rows with a missing value in any chosen column are dropped. Kendall's W cut-offs are rules of thumb.")
                except ValueError as e:
                    st.error(str(e))

# ----------------------------------------------------------- Test advisor
if nav == "Test advisor":
    st.subheader("Step 1: Does the data follow a normal distribution?")
    mode = st.radio("How do you want to decide?", ["Check my dataset automatically", "I will choose myself"], key="adv_mode")
    if mode.startswith("Check"):
        alpha = st.number_input("Significance level for the normality test", 0.01, 0.10, 0.05, 0.01, key="adv_alpha")
        thr = st.slider("Dataset counts as normal if at least this % of numeric variables are normal-like", 50, 100, 80, key="adv_thr")
        nt = cached_norm(df, alpha)
        if nt.empty:
            st.warning("No numeric variables to assess. Choose manually, or pick the categorical goal below.")
            normal = st.radio("Assume the data is", ["Normally distributed", "Not normally distributed"], key="adv_fallback") == "Normally distributed"
        else:
            st.dataframe(nt)
            normal, share = overall(nt, thr)
            if normal is None:
                normal = False
            st.markdown(f"**{share:.0f}%** of the assessed numeric variables look normal-like. Rough screen only: "
                        f"**{'mostly normal-like, so parametric tests are a reasonable starting point' if normal else 'mostly not normal-like, so non-parametric tests are the safer starting point'}**.")
            st.info("This whole-dataset summary is a rough guide. t-tests and ANOVA need normality inside each group, and regression needs it in the residuals: "
                    "use the group check below and the diagnostics in the Regression tab.")
            st.caption("Up to 200 values the Shapiro-Wilk p-value decides. With more data, tiny deviations always get flagged, so skewness and "
                       "kurtosis are used instead (rules of thumb). Check the EDA tab Q-Q plots before trusting this.")
    else:
        normal = st.radio("My data is", ["Normally distributed", "Not normally distributed"], key="adv_manual") == "Normally distributed"
    with st.expander("Better check: normality inside each group (what t-tests and ANOVA need)"):
        nc_, cc_ = E.split_columns(df)
        gc_ = [c for c in cc_ if 2 <= df[c].nunique(dropna=True) <= 10]
        if not nc_ or not gc_:
            st.write("Needs one numeric column and one grouping column with 2 to 10 groups.")
        else:
            gv = st.selectbox("Numeric variable", nc_, key="adv_gv"); gg = st.selectbox("Grouping column", gc_, key="adv_gg")
            gt = group_normality(df, gv, gg, st.session_state.get("adv_alpha", 0.05)); st.dataframe(gt)
            judged = gt[gt.verdict != "Not assessed"]
            if len(judged):
                st.markdown("Every assessed group looks normal-like: parametric tests are reasonable." if (judged.verdict == "Normal-like").all()
                            else "At least one group does not look normal-like: prefer the non-parametric test, or check the Q-Q plots (EDA tab) and the group sizes.")
            st.caption("With about 30 or more values per group, t-tests and ANOVA tolerate mild non-normality. Choose the route in Step 1 (\"I will choose myself\") to follow this result.")
    st.subheader("Step 2: What do you want to do?")
    goal = st.selectbox("Goal", GOALS, key="adv_goal")
    rec = recommend(goal, normal)
    st.success(f"{rec['family']} route. Recommended test: **{rec['test']}**")
    st.table(pd.DataFrame({"": ["Alternatives", "Post-hoc / follow-up", "Check first", "In this app"],
                           "Suggestion": [rec["alt"], rec["post"], rec["check"], rec["where"]]}))
    st.caption("Normality matters for each group (or the model residuals), not for the dataset as a whole. With about 30 or more values per group, "
               "t-tests and ANOVA tolerate mild non-normality, while heavy skew, outliers or small groups favour non-parametric tests. "
               "Deciding the test from a normality test first can itself affect error rates, so also look at plots.")

# ---------------------------------------------------------------- ANOVA
if nav == "ANOVA":
    nc, cc = E.split_columns(df)
    gcols = [c for c in cc if 3 <= df[c].nunique() <= 10]
    if not nc or not gcols:
        st.warning("Need a numeric column and a grouping column with 3 to 10 groups (use the Two-group test for 2 groups).")
    else:
        av = st.selectbox("Numeric column", nc, key="an_v")
        ag = st.selectbox("Grouping column (3 to 10 groups)", gcols, key="an_g")
        if st.button("Run ANOVA"):
            d = df[[av, ag]].dropna()
            r = multi_group_test({str(k): s[av].values for k, s in d.groupby(ag)})
            if r["k"] < 3:
                st.error("Fewer than 3 groups have at least 2 observations.")
            else:
                st.dataframe(r["groups"]); st.table(r["tests"])
                st.markdown("**Pairwise comparisons (Welch t-tests, Holm-adjusted)**"); st.dataframe(r["pairs"])
                st.markdown("**Interpretation**")
                for line in interpret_anova(r):
                    st.markdown("- " + line)
                st.caption("Welch ANOVA is the default because it does not assume equal variances. Effect-size cut-offs are rules of thumb.")

# ------------------------------------------------------------ Chi-square
if nav == "Chi-square":
    _, cc = E.split_columns(df)
    cc = [c for c in cc if df[c].nunique() >= 2]
    if len(cc) < 2:
        st.warning("Need two categorical columns.")
    else:
        c1_, c2_ = st.columns(2)
        ra = c1_.selectbox("Row variable", cc, key="ch_a")
        rb = c2_.selectbox("Column variable", cc, index=1, key="ch_b")
        if st.button("Run chi-square test"):
            if ra == rb:
                st.error("Choose two different columns.")
            else:
                d = df[[ra, rb]].dropna(); r = chi_square_test(d[ra], d[rb])
                st.write("Observed counts"); st.dataframe(r["table"])
                st.write("Adjusted standardised residuals (a value beyond +/-2 marks a cell that departs from independence)"); st.dataframe(r["residuals"])
                st.markdown("**Interpretation**")
                for line in interpret_chi(r):
                    st.markdown("- " + line)
                st.caption("Association is not causation. Cramer's V cut-offs are rules of thumb.")

# ------------------------------------------------------------------ Report
if nav == "Report":
    st.write("Build one report from your data and download it as Excel, PDF or Word.")
    r1, r2 = st.columns(2)
    rp_title = r1.text_input("Report title", "Data analysis report", key="rp_title")
    rp_author = r2.text_input("Prepared by (optional)", "", key="rp_author")
    inc = st.multiselect("Sections to include", RP.SECTIONS, default=RP.SECTIONS, key="rp_inc")
    maxv = st.slider("Variables shown in the distributions section", 1, 12, 6, key="rp_maxv")
    nc, cc = E.split_columns(df); cfg = {"title": rp_title, "author": rp_author, "include": inc, "max_vars": maxv}
    cfg["pca_log"] = st.checkbox("PCA section: apply log(1+x) to skewed non-negative variables", key="rp_pca_log")
    rcat_opts = [c for c in cc if 2 <= df[c].nunique(dropna=True) <= 10]
    if len(nc) >= 2 and st.checkbox("Add a regression", key="rp_reg_on"):
        ry = st.selectbox("Response (Y)", nc, key="rp_ry"); rx = st.multiselect("Numeric predictor(s) (X)", [c for c in nc if c != ry], default=[c for c in nc if c != ry][:1], key="rp_rx")
        rcat = st.multiselect("Categorical predictor(s) (optional, dummy-coded)", rcat_opts, key="rp_rcat")
        q1, q2 = st.columns(2)
        cfg["reg_dw"] = q1.checkbox("Rows are in time order (include Durbin-Watson)", key="rp_rdw")
        cfg["reg_ho"] = q2.checkbox("Include a hold-out test (25% of rows)", key="rp_rho")
        if rx or rcat:
            cfg["reg"] = (ry, rx); cfg["reg_cat"] = rcat
    gcols = [c for c in cc if 2 <= df[c].nunique() <= 10]
    if nc and gcols and st.checkbox("Add a group comparison (t-test or ANOVA)", key="rp_grp_on"):
        cfg["grp"] = (st.selectbox("Numeric variable", nc, key="rp_gn"), st.selectbox("Group column", gcols, key="rp_gc"))
    ccols = [c for c in cc if df[c].nunique() >= 2]
    if len(ccols) >= 2 and st.checkbox("Add a chi-square test", key="rp_chi_on"):
        ca = st.selectbox("Row variable", ccols, key="rp_ca"); cb = st.selectbox("Column variable", [c for c in ccols if c != ca], key="rp_cb")
        cfg["chi"] = (ca, cb)
    if len(nc) >= 2 and st.checkbox("Add a paired test (two related columns)", key="rp_pr_on"):
        pa_ = st.selectbox("First measurement", nc, key="rp_pa"); pb_ = st.selectbox("Second measurement", [c for c in nc if c != pa_], key="rp_pb")
        cfg["paired"] = (pa_, pb_)
    if nc and st.checkbox("Add a one-sample test", key="rp_os_on"):
        oc_ = st.selectbox("Numeric column", nc, key="rp_oc"); om_ = st.number_input("Value to compare with", value=0.0, key="rp_omu")
        cfg["one"] = (oc_, float(om_))
    if len(nc) >= 3 and st.checkbox("Add a Friedman test (3 or more related columns)", key="rp_fr_on"):
        fcs_ = st.multiselect("Related columns", nc, default=nc[:3], key="rp_fcols")
        if len(fcs_) >= 3:
            cfg["friedman"] = fcs_
    if st.button("Build report", key="rp_go"):
        with st.spinner("Building the report (this can take a few seconds)..."):
            blocks = RP.build_report(df, fname, cfg)
            st.session_state["report"] = {"source": fname, "xlsx": RP.to_xlsx(blocks), "docx": RP.to_docx(blocks), "pdf": RP.to_pdf(blocks)}
    rep = st.session_state.get("report")
    if rep and rep["source"] == fname:
        st.success("Report ready. Download it in the format you need:")
        base = fname.rsplit(".", 1)[0]; d1, d2, d3 = st.columns(3)
        d1.download_button("Download Excel", rep["xlsx"], f"report_{base}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_x")
        d2.download_button("Download PDF", rep["pdf"], f"report_{base}.pdf", "application/pdf", key="dl_p")
        d3.download_button("Download Word", rep["docx"], f"report_{base}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="dl_w")
        st.caption("Tables longer than 40 rows are shortened in the report. Interpretations are automatic rules of thumb.")

theme.footer()
