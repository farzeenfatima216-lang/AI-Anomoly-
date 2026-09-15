import io
import os
import pickle

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


MODEL_PATH = os.path.join("models", "isolation_forest.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
RANDOM_STATE = 42


st.set_page_config(
    page_title="AI Anomaly Detector",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #09090b;
        --panel: #131316;
        --panel-light: #1c1b20;
        --line: #30272b;
        --maroon: #9e2949;
        --maroon-bright: #d04a68;
        --text: #f5eef0;
        --muted: #a99da2;
    }

    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: var(--ink); color: var(--text); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #101013; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] *,
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #1a171b;
        border-color: #713047;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #ffffff !important;
    }
    h1, h2, h3 { letter-spacing: -0.03em; }
    h1 { font-size: clamp(2.2rem, 4vw, 4.4rem) !important; line-height: .98 !important; }
    h2 { margin-top: 1.8rem !important; }
    .hero { padding: 2.2rem 0 1.2rem; border-bottom: 1px solid var(--line); margin-bottom: 1.3rem; }
    .eyebrow { color: var(--maroon-bright); font: 500 .76rem 'DM Mono', monospace; letter-spacing: .13em; text-transform: uppercase; }
    .hero p { color: var(--muted); max-width: 640px; font-size: 1.02rem; }
    .brand { border-left: 3px solid var(--maroon); padding-left: 1rem; margin-bottom: 2rem; }
    .brand-title { font-weight: 700; font-size: 1.25rem; }
    .brand-copy { color: var(--muted); font-size: .84rem; line-height: 1.45; margin-top: .4rem; }
    .kpi { background: linear-gradient(135deg, var(--panel-light), var(--panel)); border: 1px solid var(--line); border-top: 2px solid var(--maroon); padding: 1rem 1.1rem; min-height: 106px; }
    .kpi-label { color: var(--muted); font: 500 .72rem 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .08em; }
    .kpi-value { color: var(--text); font-size: 1.85rem; font-weight: 700; margin-top: .45rem; }
    .kpi-note { color: var(--muted); font-size: .75rem; }
    .insight { background: #151216; border: 1px solid var(--line); border-left: 3px solid var(--maroon); padding: .85rem 1rem; min-height: 86px; }
    .insight-title { color: var(--maroon-bright); font: 500 .7rem 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .08em; }
    .insight-copy { color: var(--text); font-size: .88rem; line-height: 1.35; margin-top: .4rem; }
    .status-pill { display: inline-flex; align-items: center; gap: .45rem; padding: .35rem .65rem; border: 1px solid #713047; background: #29131b; color: #f3b9c6; font: 500 .72rem 'DM Mono', monospace; letter-spacing: .06em; }
    .status-dot { color: var(--maroon-bright); font-size: 1rem; line-height: 0; }
    .panel-title { color: var(--text); font-size: 1.15rem; font-weight: 600; margin: 1.3rem 0 .25rem; }
    .panel-copy { color: var(--muted); font-size: .86rem; margin-bottom: .8rem; }
    .footer { border-top: 1px solid var(--line); color: var(--muted); margin-top: 2rem; padding: 1.4rem 0 2rem; font: .72rem 'DM Mono', monospace; display: flex; justify-content: space-between; gap: 1rem; }
    .section-note { color: var(--muted); margin: -.5rem 0 1rem; }
    .notice { background: #21161a; border: 1px solid #5e2a39; color: #f0cbd3; padding: .8rem 1rem; margin: .6rem 0 1rem; }
    .stButton > button, .stDownloadButton > button { border: 1px solid #713047; background: #461726; color: #fff5f7; border-radius: 3px; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--maroon-bright); background: var(--maroon); color: white; }
    [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); padding: .7rem; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); }
    .mono { font-family: 'DM Mono', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)


PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = {
    "template": PLOTLY_TEMPLATE,
    "paper_bgcolor": "#131316",
    "plot_bgcolor": "#131316",
    "font": {"color": "#ffffff", "family": "Space Grotesk"},
    "title_font": {"color": "#ffffff", "size": 18},
    "legend": {"font": {"color": "#ffffff", "size": 13}},
    "xaxis": {
        "title_font": {"color": "#ffffff"},
        "tickfont": {"color": "#ffffff"},
        "linecolor": "#706168",
    },
    "yaxis": {
        "title_font": {"color": "#ffffff"},
        "tickfont": {"color": "#ffffff"},
        "linecolor": "#706168",
    },
    "colorway": ["#d04a68", "#9e2949", "#c49aa5", "#706168"],
    "margin": {"l": 30, "r": 20, "t": 55, "b": 35},
}


def load_pickle(path):
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as artifact:
                return pickle.load(artifact)
        except Exception:
            return None


def clean_numeric_features(dataframe):
    feature_frame = dataframe.drop(columns=["Class"], errors="ignore").copy()
    numeric_frame = feature_frame.select_dtypes(include=np.number).copy()
    medians = numeric_frame.median(numeric_only=True)
    return numeric_frame.fillna(medians), numeric_frame.columns.tolist()


def artifact_schema_matches(model, scaler, columns):
    model_columns = list(getattr(model, "feature_names_in_", []))
    scaler_columns = list(getattr(scaler, "feature_names_in_", []))
    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and expected_count != len(columns):
        return False
    if model_columns and model_columns != columns:
        return False
    if scaler_columns and scaler_columns != columns:
        return False
    return True


def run_detection(dataframe, contamination):
    feature_frame, columns = clean_numeric_features(dataframe)
    if feature_frame.empty or not columns:
        raise ValueError("The CSV has no usable numerical feature columns.")

    saved_model = load_pickle(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    saved_scaler = load_pickle(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    can_reuse = (
        saved_model is not None
        and saved_scaler is not None
        and artifact_schema_matches(saved_model, saved_scaler, columns)
        and abs(float(getattr(saved_model, "contamination", contamination)) - contamination) < 1e-9
    )

    if can_reuse:
        model = saved_model
        scaler = saved_scaler
        scaled_features = scaler.transform(feature_frame)
        source = "Saved model and scaler"
    else:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_frame)
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(scaled_features)
        source = "Fresh model fitted for this upload"

    raw_predictions = model.predict(scaled_features)
    anomaly_score = -model.decision_function(scaled_features)
    results = dataframe.copy()
    results["Anomaly_Score"] = anomaly_score
    results["Status"] = np.where(raw_predictions == -1, "Anomaly", "Normal")
    results["Model_Prediction"] = raw_predictions
    return results, feature_frame, model, scaler, source


def chart_layout(figure):
    figure.update_layout(**PLOTLY_LAYOUT)
    return figure


def kpi_card(label, value, note=""):
    return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>'


def insight_card(title, copy):
    return f'<div class="insight"><div class="insight-title">{title}</div><div class="insight-copy">{copy}</div></div>'


def safe_sample(dataframe, size=10000):
    if len(dataframe) <= size:
        return dataframe
    return dataframe.sample(size, random_state=RANDOM_STATE)


def add_time_bucket(dataframe):
    if "Time" not in dataframe.columns:
        return None
    time_frame = dataframe[["Time", "Status"]].copy()
    time_frame["Time_Bucket"] = pd.cut(
        time_frame["Time"], bins=40, include_lowest=True
    ).astype(str)
    return time_frame.groupby(["Time_Bucket", "Status"], observed=False).size().reset_index(name="Count")


def plot_amount_charts(results, key_prefix="amount"):
    if "Amount" not in results.columns:
        return
    amount_frame = safe_sample(results)
    left, right = st.columns(2)
    with left:
        figure = px.histogram(
            amount_frame,
            x="Amount",
            color="Status",
            marginal="box",
            title="Transaction Amount Distribution",
            color_discrete_map={"Normal": "#706168", "Anomaly": "#d04a68"},
        )
        st.plotly_chart(
            chart_layout(figure), use_container_width=True,
            key=f"{key_prefix}_distribution",
        )
    with right:
        figure = px.box(
            amount_frame,
            x="Status",
            y="Amount",
            color="Status",
            points=False,
            title="Transaction Amount: Normal vs Anomaly",
            color_discrete_map={"Normal": "#706168", "Anomaly": "#d04a68"},
        )
        st.plotly_chart(
            chart_layout(figure), use_container_width=True,
            key=f"{key_prefix}_boxplot",
        )


def plot_time_activity(results, key="time_activity"):
    if results["Time"].nunique(dropna=True) < 2:
        st.info("Time data does not contain enough variation for a time-series chart.")
        return
    time_frame = add_time_bucket(results)
    if time_frame is None:
        return
    figure = px.area(
        time_frame,
        x="Time_Bucket",
        y="Count",
        color="Status",
        title="Transactions Over Time",
        color_discrete_map={"Normal": "#706168", "Anomaly": "#d04a68"},
    )
    figure.update_xaxes(type="category", nticks=10)
    st.plotly_chart(
        chart_layout(figure), use_container_width=True, key=key
    )


def show_quality(dataframe):
    numeric_columns = dataframe.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = dataframe.select_dtypes(exclude=np.number).columns.tolist()
    left, right = st.columns(2)
    with left:
        st.markdown("**Missing values by column**")
        st.dataframe(dataframe.isna().sum().rename("Missing Values"), use_container_width=True)
        st.markdown("**Data types**")
        st.dataframe(dataframe.dtypes.astype(str).rename("Data Type"), use_container_width=True)
    with right:
        st.markdown("**Feature groups**")
        st.write({"Numerical columns": numeric_columns, "Categorical columns": categorical_columns})
        st.metric("Duplicate rows", int(dataframe.duplicated().sum()))
        st.metric("Total missing values", int(dataframe.isna().sum().sum()))


def show_evaluation(results):
    if "Class" not in results.columns:
        st.info("Ground-truth labels are not available, so supervised evaluation metrics cannot be calculated.")
        return
    labels = pd.to_numeric(results["Class"], errors="coerce")
    valid = labels.isin([0, 1])
    if not valid.any() or valid.sum() != len(results):
        st.warning("A Class column exists, but it does not contain only valid 0/1 ground-truth labels.")
        return
    y_true = labels.astype(int)
    y_pred = (results["Status"] == "Anomaly").astype(int)
    scores = results["Anomaly_Score"]
    if y_true.nunique() < 2:
        st.warning(
            "Evaluation requires both Normal and Fraud ground-truth labels."
        )
        return
    metrics = {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, scores),
        "PR-AUC": average_precision_score(y_true, scores),
    }
    st.markdown('<div class="notice">The Class column is used only as ground truth for evaluation and is not used as an input feature.</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for column, (name, value) in zip(cols, metrics.items()):
        column.metric(name, f"{value:.3f}")
    st.dataframe(pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T, use_container_width=True)
    matrix = confusion_matrix(y_true, y_pred)
    fig = px.imshow(matrix, text_auto=True, x=["Normal", "Fraud"], y=["Normal", "Fraud"], color_continuous_scale=[[0, "#25151b"], [1, "#d04a68"]], title="Confusion Matrix")
    st.plotly_chart(
        chart_layout(fig), use_container_width=True, key="evaluation_confusion_matrix"
    )


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-title">AI Anomaly Detector</div>'
        '<div class="brand-copy">Credit Card Transaction Monitoring</div></div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("**Detection settings**")
    contamination = st.slider(
        "Contamination", 0.01, 0.10, 0.02, 0.01
    )
    st.markdown("---")
    st.markdown("**Model**")
    st.markdown("Isolation Forest")
    st.caption("Unsupervised anomaly detection")
    st.markdown("---")
    st.markdown("**Dataset status**")
    if uploaded_file is None:
        st.caption("Not uploaded")
    else:
        st.caption("Uploaded · ready for analysis")
    if st.button("Clear data", use_container_width=True):
        st.rerun()


st.markdown(
    '<div class="hero"><div class="eyebrow">Financial intelligence / 01</div>'
    '<h1>AI ANOMALY DETECTOR</h1>'
    '<p><strong>Intelligent Credit Card Transaction Monitoring</strong></p>'
    '<p>Detect unusual transaction patterns using Isolation Forest and review '
    'suspicious activity through interactive analytics.</p>'
    '<span class="status-pill"><span class="status-dot">●</span> MODEL ACTIVE '
    '· ISOLATION FOREST</span></div>',
    unsafe_allow_html=True,
)

if uploaded_file is None:
    st.markdown('<div class="notice">Upload a CSV from the sidebar to begin analysis. The standard schema is Time, V1-V28, Amount, and optional Class.</div>', unsafe_allow_html=True)
    st.stop()

try:
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        st.error("The uploaded CSV is empty.")
        st.stop()
    dataframe = pd.read_csv(io.BytesIO(raw_bytes))
except Exception as error:
    st.error(f"The CSV could not be read: {error}")
    st.stop()

if dataframe.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

with st.sidebar:
    st.caption(f"Rows · {len(dataframe):,}")
    st.caption(f"Columns · {len(dataframe.columns):,}")

try:
    with st.spinner("Running anomaly detection..."):
        results, feature_frame, fitted_model, fitted_scaler, model_source = run_detection(dataframe, contamination)
except ValueError as error:
    st.error(str(error))
    st.stop()
except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

anomaly_mask = results["Status"].eq("Anomaly")
anomaly_count = int(anomaly_mask.sum())
normal_count = int((~anomaly_mask).sum())
anomaly_rate = anomaly_count / len(results) * 100

st.markdown("### Current run")
amount_average = results["Amount"].mean() if "Amount" in results.columns else None
amount_maximum = results["Amount"].max() if "Amount" in results.columns else None
card_columns = st.columns(6)
kpi_values = [
    ("Total transactions", f"{len(results):,}", "Uploaded rows"),
    ("Anomalies detected", f"{anomaly_count:,}", "Potential anomalies"),
    ("Normal transactions", f"{normal_count:,}", "Within learned pattern"),
    ("Anomaly rate", f"{anomaly_rate:.2f}%", f"Threshold {contamination:.2f}"),
    ("Average amount", f"{amount_average:,.2f}" if amount_average is not None else "N/A", "Across uploaded data"),
    ("Highest amount", f"{amount_maximum:,.2f}" if amount_maximum is not None else "N/A", "Largest transaction"),
]
for column, values in zip(card_columns, kpi_values):
    column.markdown(kpi_card(*values), unsafe_allow_html=True)
st.caption(
    f"{model_source}. Class is excluded from model inputs. "
    f"Numerical features used: {len(feature_frame.columns)}."
)

st.markdown('<div class="panel-title">Quick Insights</div>', unsafe_allow_html=True)
insight_columns = st.columns(4)
insights = [
    ("Anomaly rate", f"{anomaly_rate:.2f}% of transactions were flagged as unusual."),
    ("Transaction volume", f"{len(results):,} transactions analyzed in this run."),
    ("Average amount", f"Average transaction amount is {amount_average:,.2f}." if amount_average is not None else "Amount data is unavailable."),
    ("Detection status", f"Isolation Forest identified {anomaly_count:,} suspicious records."),
]
for column, values in zip(insight_columns, insights):
    column.markdown(insight_card(*values), unsafe_allow_html=True)

overview_tab, suspicious_tab, visual_tab, quality_tab, evaluation_tab = st.tabs([
    "Overview", "Suspicious Records", "Visual Analysis", "Data Quality", "Model Evaluation"
])

with overview_tab:
    st.subheader("Transaction Analytics")
    st.markdown(
        '<div class="section-note">The detector highlights records that differ from '
        'the dominant transaction pattern. Review signals with business context.</div>',
        unsafe_allow_html=True,
    )
    summary = pd.DataFrame(
        {"Status": ["Normal", "Anomaly"], "Count": [normal_count, anomaly_count]}
    )
    chart_left, chart_right = st.columns(2)
    with chart_left:
        figure = px.bar(
            summary,
            x="Status",
            y="Count",
            color="Status",
            title="Normal vs Anomaly",
            color_discrete_map={"Normal": "#706168", "Anomaly": "#d04a68"},
        )
        st.plotly_chart(
            chart_layout(figure), use_container_width=True,
            key="overview_status_bar",
        )
    with chart_right:
        figure = px.histogram(
            safe_sample(results),
            x="Anomaly_Score",
            color="Status",
            title="Anomaly Score Distribution",
            color_discrete_map={"Normal": "#706168", "Anomaly": "#d04a68"},
        )
        st.plotly_chart(
            chart_layout(figure), use_container_width=True,
            key="overview_score_histogram",
        )
    if "Amount" in results.columns:
        plot_amount_charts(results, key_prefix="overview_amount")
    if "Time" in results.columns:
        plot_time_activity(results, key="overview_time_activity")
    st.markdown('<div class="panel-title">Anomaly Intelligence</div>', unsafe_allow_html=True)
    intelligence_columns = st.columns(5)
    intelligence_values = [
        ("Anomaly count", f"{anomaly_count:,}"),
        ("Anomaly percentage", f"{anomaly_rate:.2f}%"),
        ("Average score", f"{results['Anomaly_Score'].mean():.4f}"),
        ("Minimum score", f"{results['Anomaly_Score'].min():.4f}"),
        ("Maximum score", f"{results['Anomaly_Score'].max():.4f}"),
    ]
    for column, values in zip(intelligence_columns, intelligence_values):
        column.markdown(kpi_card(*values), unsafe_allow_html=True)
    st.caption(
        "Lower anomaly scores indicate observations that are more unusual relative "
        "to the learned data pattern. Anomalies are not confirmed fraud."
    )

with suspicious_tab:
    st.subheader("Top Suspicious Transactions")
    st.markdown(
        '<div class="section-note">Higher anomaly scores indicate records that are '
        'more unusual according to the fitted model.</div>',
        unsafe_allow_html=True,
    )
    filter_left, filter_mid, filter_right = st.columns([1, 1, 1])
    with filter_left:
        status_filter = st.selectbox("Status", ["All", "Anomaly", "Normal"])
    with filter_mid:
        score_min = float(results["Anomaly_Score"].min())
        score_max = float(results["Anomaly_Score"].max())
        if score_min == score_max:
            score_filter = score_min
            st.caption(f"Minimum anomaly score · {score_min:.4f}")
        else:
            score_filter = st.slider(
                "Minimum anomaly score", score_min, score_max, score_min
            )
    with filter_right:
        record_limit = st.selectbox(
            "Show Top N Suspicious Records", [10, 25, 50, 100], index=1
        )
    filtered = results[results["Anomaly_Score"] >= score_filter].copy()
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]
    if "Amount" in filtered.columns and not filtered.empty:
        amount_min, amount_max = float(filtered["Amount"].min()), float(filtered["Amount"].max())
        if amount_min == amount_max:
            amount_filter = (amount_min, amount_max)
            st.caption(f"Amount range · {amount_min:,.2f}")
        else:
            amount_filter = st.slider(
                "Amount range", amount_min, amount_max,
                (amount_min, amount_max)
            )
            filtered = filtered[filtered["Amount"].between(*amount_filter)]
    filtered = filtered.sort_values("Anomaly_Score", ascending=False).head(record_limit)
    st.dataframe(filtered, use_container_width=True, height=460)
    download_data = results[anomaly_mask].sort_values("Anomaly_Score", ascending=False).to_csv(index=False).encode("utf-8")
    st.download_button("Download Anomalies CSV", download_data, "detected_anomalies.csv", "text/csv")

with visual_tab:
    st.subheader("Visual Analysis")
    st.markdown(
        '<div class="section-note">Interactive charts use sampled records for '
        'display performance only. Detection and exports always use the full dataset.</div>',
        unsafe_allow_html=True,
    )
    col_one, col_two = st.columns(2)
    with col_one:
        distribution = results["Status"].value_counts().rename_axis("Status").reset_index(name="Count")
        figure = px.pie(
            distribution,
            names="Status",
            values="Count",
            hole=0.58,
            title="Normal vs Anomaly Distribution",
            color_discrete_map={"Normal": "#706168", "Anomaly": "#d04a68"},
        )
        st.plotly_chart(
            chart_layout(figure), use_container_width=True,
            key="visual_status_pie",
        )
    with col_two:
        figure = px.histogram(
            safe_sample(results),
            x="Anomaly_Score",
            color="Status",
            title="Anomaly Score Distribution",
            color_discrete_map={"Normal": "#706168", "Anomaly": "#d04a68"},
        )
        st.plotly_chart(
            chart_layout(figure), use_container_width=True,
            key="visual_score_histogram",
        )
    if "Amount" in results.columns:
        plot_amount_charts(results, key_prefix="visual_amount")
    if "Time" in results.columns:
        plot_time_activity(results, key="visual_time_activity")
    suspicious_chart = results[results["Status"] == "Anomaly"].nlargest(
        15, "Anomaly_Score"
    ).sort_values("Anomaly_Score")
    if not suspicious_chart.empty:
        figure = px.bar(
            suspicious_chart,
            x="Anomaly_Score",
            y=suspicious_chart.index.astype(str),
            orientation="h",
            title="Top Suspicious Transactions",
            labels={"y": "Record", "x": "Anomaly Score"},
            color_discrete_sequence=["#d04a68"],
        )
        st.plotly_chart(
            chart_layout(figure), use_container_width=True,
            key="visual_suspicious_bar",
        )
    if len(feature_frame.columns) >= 2:
        plot_features = safe_sample(feature_frame)
        plot_labels = results.loc[plot_features.index, "Status"]
        if len(feature_frame.columns) == 2:
            coordinates = plot_features.to_numpy()
            axis_names = list(plot_features.columns)
        else:
            coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(plot_features)
            axis_names = ["PC1", "PC2"]
        pca_frame = pd.DataFrame(
            {
                axis_names[0]: coordinates[:, 0],
                axis_names[1]: coordinates[:, 1],
                "Status": plot_labels.values,
            }
        )
        figure = px.scatter(
            pca_frame,
            x=axis_names[0],
            y=axis_names[1],
            color="Status",
            title="2D Feature Projection",
            color_discrete_map={"Normal": "#706168", "Anomaly": "#d04a68"},
        )
        st.plotly_chart(
            chart_layout(figure), use_container_width=True,
            key="visual_pca_scatter",
        )
    else:
        st.info("A 2D visualization requires at least two numerical features.")

with quality_tab:
    st.subheader("Data Quality")
    metric_columns = st.columns(5)
    quality_values = [
        ("Missing values", int(dataframe.isna().sum().sum())),
        ("Duplicate rows", int(dataframe.duplicated().sum())),
        ("Numerical features", len(dataframe.select_dtypes(include=np.number).columns)),
        ("Categorical features", len(dataframe.select_dtypes(exclude=np.number).columns)),
        ("Dataset size", f"{len(dataframe):,} rows"),
    ]
    for column, (label, value) in zip(metric_columns, quality_values):
        column.markdown(kpi_card(label, value), unsafe_allow_html=True)
    show_quality(dataframe)
    missing_frame = dataframe.isna().sum().reset_index()
    missing_frame.columns = ["Column", "Missing Values"]
    missing_frame = missing_frame[missing_frame["Missing Values"] > 0]
    if not missing_frame.empty:
        figure = px.bar(
            missing_frame,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column",
            color_discrete_sequence=["#d04a68"],
        )
        st.plotly_chart(
            chart_layout(figure), use_container_width=True,
            key="quality_missing_values",
        )
    st.markdown("**Dataset preview**")
    st.dataframe(dataframe.head(10), use_container_width=True)

with evaluation_tab:
    st.subheader("Ground-truth evaluation")
    show_evaluation(results)

st.markdown(
    '<div class="footer"><span>AI Anomaly Detector · Isolation Forest · '
    'Credit Card Transaction Analysis</span><span>Built with Python, '
    'Scikit-learn, Plotly &amp; Streamlit</span></div>',
    unsafe_allow_html=True,
)
