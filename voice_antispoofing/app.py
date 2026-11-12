
from pathlib import Path
import json

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

TRAINING_FEATURES = RESULTS / "training_features.csv"
VALIDATION_METRICS = RESULTS / "svm_tuned_validation_metrics.json"
DEV_METRICS = RESULTS / "svm_dev_metrics.json"
DEV_PREDICTIONS = RESULTS / "svm_dev_predictions.csv"
THRESHOLD_ANALYSIS = RESULTS / "svm_threshold_analysis.csv"


st.set_page_config(
    page_title="VoiceGuard | SVM Anti-Spoofing",
    page_icon="🛡️",
    layout="wide",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0b1220, #111d30);
        color: #edf4ff;
    }

    [data-testid="stMetric"] {
        background: #18263a;
        border: 1px solid #2a405b;
        padding: 16px;
        border-radius: 12px;
    }

    h1, h2, h3 {
        color: #67e8d2 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛡️ VoiceGuard")
st.caption("TUNED SVM-BASED VOICE ANTI-SPOOFING | RESEARCH PROTOTYPE")

st.markdown(
    """
    VoiceGuard analyses audio recordings to distinguish genuine
    speech from spoofed speech using MFCC features and a tuned
    Support Vector Machine classifier.
    """
)

st.divider()


# --------------------------------------------------
# Load files
# --------------------------------------------------

required_files = [
    TRAINING_FEATURES,
    VALIDATION_METRICS,
    DEV_METRICS,
    DEV_PREDICTIONS,
    THRESHOLD_ANALYSIS,
]

missing_files = [file.name for file in required_files if not file.exists()]

if missing_files:
    st.error("Required result files are missing:")
    for filename in missing_files:
        st.write(f"- {filename}")
    st.stop()


try:
    training_features = pd.read_csv(TRAINING_FEATURES)
    dev_predictions = pd.read_csv(DEV_PREDICTIONS)
    threshold_data = pd.read_csv(THRESHOLD_ANALYSIS)

    with open(VALIDATION_METRICS, "r", encoding="utf-8") as file:
        validation_metrics = json.load(file)

    with open(DEV_METRICS, "r", encoding="utf-8") as file:
        dev_metrics = json.load(file)

except Exception as error:
    st.error(f"Unable to load project results: {error}")
    st.stop()


# --------------------------------------------------
# Validate columns
# --------------------------------------------------

if not {"filename", "label"}.issubset(training_features.columns):
    st.error("Training features CSV is missing required columns.")
    st.stop()

prediction_columns = {
    "filename",
    "actual_label",
    "predicted_label",
    "spoof_decision_score",
}

if not prediction_columns.issubset(dev_predictions.columns):
    st.error("Development predictions CSV is missing required columns.")
    st.stop()

threshold_columns = {
    "threshold",
    "accuracy",
    "FAR",
    "FRR",
    "average_error_rate",
}

if not threshold_columns.issubset(threshold_data.columns):
    st.error("Threshold analysis CSV is missing required columns.")
    st.stop()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def render_confusion_matrix(metrics):
    labels = metrics["confusion_matrix_labels"]

    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"Actual {label.title()}" for label in labels],
        columns=[f"Predicted {label.title()}" for label in labels],
    )

    st.dataframe(matrix, use_container_width=True)


def render_classification_report(metrics):
    report = metrics["classification_report"]

    rows = []

    for label in ["genuine", "spoof"]:
        values = report[label]

        rows.append({
            "Class": label.title(),
            "Precision": f"{values['precision'] * 100:.2f}%",
            "Recall": f"{values['recall'] * 100:.2f}%",
            "F1-score": f"{values['f1-score'] * 100:.2f}%",
            "Support": int(values["support"]),
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
    )


def render_evaluation(title, metrics):
    st.subheader(title)

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    c2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    c3.metric("Approximate EER", f"{metrics['eer_approximate'] * 100:.2f}%")

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "False Acceptance Rate",
        f"{metrics['far_spoof_accepted_as_genuine'] * 100:.2f}%",
    )

    c5.metric(
        "False Rejection Rate",
        f"{metrics['frr_genuine_rejected_as_spoof'] * 100:.2f}%",
    )

    c6.metric(
        "Evaluation Samples",
        f"{metrics.get('validation_samples', metrics.get('total_recordings', 0)):,}",
    )

    st.markdown("#### Confusion Matrix")
    st.caption("Rows represent actual classes; columns represent predicted classes.")

    render_confusion_matrix(metrics)

    st.markdown("#### Classification Report")
    render_classification_report(metrics)


# --------------------------------------------------
# Dataset overview
# --------------------------------------------------

st.header("1. Dataset Overview")

train_genuine = int(
    (training_features["label"] == "genuine").sum()
)
train_spoof = int(
    (training_features["label"] == "spoof").sum()
)

dev_genuine = int(
    (dev_predictions["actual_label"] == "genuine").sum()
)
dev_spoof = int(
    (dev_predictions["actual_label"] == "spoof").sum()
)

c1, c2, c3 = st.columns(3)

c1.metric("Training Recordings", f"{len(training_features):,}")
c2.metric("Development Recordings", f"{len(dev_predictions):,}")
c3.metric("Processing Errors", f"{dev_metrics['processing_errors']:,}")

left, right = st.columns(2)

with left:
    st.markdown("#### Training Dataset Distribution")

    train_distribution = pd.Series(
        {
            "Genuine": train_genuine,
            "Spoof": train_spoof,
        }
    )

    st.bar_chart(train_distribution)

with right:
    st.markdown("#### Development Dataset Distribution")

    dev_distribution = pd.Series(
        {
            "Genuine": dev_genuine,
            "Spoof": dev_spoof,
        }
    )

    st.bar_chart(dev_distribution)


st.divider()


# --------------------------------------------------
# Final algorithm
# --------------------------------------------------

st.header("2. Final Algorithm Configuration")

parameters = validation_metrics["best_parameters"]

algorithm_details = pd.DataFrame([
    {"Component": "Classifier", "Configuration": "Support Vector Machine (SVM)"},
    {"Component": "Kernel", "Configuration": parameters["kernel"].upper()},
    {"Component": "Feature Extraction", "Configuration": "13 MFCC coefficients"},
    {"Component": "Preprocessing", "Configuration": "StandardScaler"},
    {"Component": "Hyperparameter Tuning", "Configuration": "GridSearchCV"},
    {"Component": "Cross-validation", "Configuration": f"{validation_metrics['cross_validation_folds']}-fold StratifiedKFold"},
    {"Component": "Best C", "Configuration": str(parameters["C"])},
    {"Component": "Best Gamma", "Configuration": str(parameters["gamma"])},
    {"Component": "Decision Threshold", "Configuration": f"{dev_metrics['selected_operating_threshold']:.6f}"},
])

st.dataframe(
    algorithm_details,
    hide_index=True,
    use_container_width=True,
)

st.metric(
    "Best Cross-validation ROC-AUC",
    f"{validation_metrics['best_cross_validation_roc_auc']:.4f}",
)


st.divider()


# --------------------------------------------------
# Training validation
# --------------------------------------------------

st.header("3. Training Dataset Validation")

st.write(
    "Evaluation using an 80/20 stratified holdout split "
    "from the ASVspoof 2017 V2 training dataset."
)

st.caption(
    f"Training samples: {validation_metrics['training_samples']:,} | "
    f"Validation samples: {validation_metrics['validation_samples']:,}"
)

render_evaluation(
    "Tuned SVM Validation Performance",
    validation_metrics,
)


st.warning(
    "These validation results are from a holdout of the training dataset. "
    "They should not be interpreted as independent development-set performance."
)


st.divider()


# --------------------------------------------------
# Independent development evaluation
# --------------------------------------------------

st.header("4. Development Dataset Evaluation")

st.write(
    "Evaluation on the separate ASVspoof 2017 V2 development dataset."
)

st.metric(
    "Selected Operating Threshold",
    f"{dev_metrics['selected_operating_threshold']:.6f}",
)

st.caption(
    f"Threshold selection method: "
    f"{dev_metrics['threshold_selection_method']}"
)

render_evaluation(
    "Development Set Performance",
    dev_metrics,
)

st.warning(
    "The operating threshold was selected using the development dataset's "
    "equal-error-rate analysis. Therefore, these development results are "
    "model-development results, not a final blind test evaluation."
)


st.divider()


# --------------------------------------------------
# Threshold analysis
# --------------------------------------------------

st.header("5. Decision Threshold Analysis")

st.write(
    "The following chart shows how accuracy, FAR, and FRR "
    "change with different decision thresholds."
)

threshold_chart = threshold_data.set_index("threshold")[
    ["accuracy", "FAR", "FRR", "average_error_rate"]
]

st.line_chart(threshold_chart)

st.dataframe(
    threshold_data.style.format({
        "threshold": "{:.6f}",
        "accuracy": "{:.2%}",
        "FAR": "{:.2%}",
        "FRR": "{:.2%}",
        "average_error_rate": "{:.2%}",
    }),
    hide_index=True,
    use_container_width=True,
)


st.divider()


# --------------------------------------------------
# Prediction explorer
# --------------------------------------------------

st.header("6. Development Recording Prediction Explorer")

st.write(
    "Search and filter the predictions generated by the tuned SVM "
    "on the development recordings."
)

search = st.text_input(
    "Search recording filename",
    placeholder="e.g. D_1000001.wav",
)

actual_filter = st.selectbox(
    "Filter by actual class",
    ["All", "genuine", "spoof"],
)

predicted_filter = st.selectbox(
    "Filter by predicted class",
    ["All", "genuine", "spoof"],
)

filtered = dev_predictions.copy()

if search.strip():
    filtered = filtered[
        filtered["filename"].str.contains(
            search.strip(),
            case=False,
            na=False,
        )
    ]

if actual_filter != "All":
    filtered = filtered[
        filtered["actual_label"] == actual_filter
    ]

if predicted_filter != "All":
    filtered = filtered[
        filtered["predicted_label"] == predicted_filter
    ]

st.dataframe(
    filtered[
        [
            "filename",
            "actual_label",
            "predicted_label",
            "spoof_decision_score",
        ]
    ],
    hide_index=True,
    use_container_width=True,
)

st.caption(
    f"Showing {len(filtered):,} of {len(dev_predictions):,} recordings."
)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered predictions",
    data=csv_bytes,
    file_name="svm_dev_predictions_filtered.csv",
    mime="text/csv",
)


st.divider()


# --------------------------------------------------
# Project limitations
# --------------------------------------------------

st.header("7. Project Scope")

st.markdown(
    """
    **Current module:** Voice anti-spoofing using MFCC features
    and a tuned SVM classifier.

    **Current capability:** Classifies recordings as genuine or spoof.

    **Not yet implemented in this module:**
    - Speaker identity verification.
    - Face authentication.
    - Integrated face–voice fusion.
    - Complete risk-adaptive authentication.
    """
)

st.caption(
    "VoiceGuard is an academic research prototype. "
    "Performance metrics are dataset-based and do not establish "
    "real-world authentication security."
)

# Random forest
# from pathlib import Path
# import json

# import pandas as pd
# import streamlit as st

# # --------------------------------------------------
# # Configuration
# # --------------------------------------------------
# ROOT = Path(__file__).resolve().parent
# RESULTS = ROOT / "results"
# FEATURES_FILE = RESULTS / "training_features.csv"
# PREDICTIONS_FILE = RESULTS / "dataset_predictions.csv"
# METRICS_FILE = RESULTS / "validation_metrics.json"

# st.set_page_config(
#     page_title="VoiceGuard | Anti-Spoofing",
#     page_icon="🛡️",
#     layout="wide",
# )

# # --------------------------------------------------
# # Styling
# # --------------------------------------------------
# st.markdown(
#     """
#     <style>
#     .stApp {
#         background: linear-gradient(135deg, #0b1220, #111d30);
#         color: #edf4ff;
#     }
#     [data-testid="stMetric"] {
#         background: #18263a;
#         border: 1px solid #2a405b;
#         padding: 18px;
#         border-radius: 12px;
#     }
#     h1, h2, h3 {
#         color: #67e8d2 !important;
#     }
#     div[data-testid="stDataFrame"] {
#         border: 1px solid #2a405b;
#         border-radius: 10px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # --------------------------------------------------
# # Header
# # --------------------------------------------------
# st.title("🛡️ VoiceGuard")
# st.caption("VOICE ANTI-SPOOFING • RESEARCH PROTOTYPE")
# st.markdown(
#     "Explore the ASVspoof 2017 V2 training dataset, "
#     "review model validation results, and inspect recording predictions."
# )

# st.divider()

# # --------------------------------------------------
# # Load existing files
# # --------------------------------------------------
# if not FEATURES_FILE.exists():
#     st.error(f"Feature file not found: {FEATURES_FILE}")
#     st.info("Run scripts/prepare_features.py first.")
#     st.stop()

# if not PREDICTIONS_FILE.exists():
#     st.error(f"Prediction file not found: {PREDICTIONS_FILE}")
#     st.info("Run scripts/predict_dataset.py first.")
#     st.stop()

# features = pd.read_csv(FEATURES_FILE)
# predictions = pd.read_csv(PREDICTIONS_FILE)

# # Load metrics generated by train_model.py
# metrics = None

# if METRICS_FILE.exists():
#     with open(METRICS_FILE, "r", encoding="utf-8") as file:
#         metrics = json.load(file)
# else:
#     st.warning(
#         "Validation metrics file not found. "
#         "Run scripts/train_model.py to generate it."
#     )

# required_columns = {"filename", "label"}
# if not required_columns.issubset(features.columns):
#     st.error("The feature CSV is missing filename or label columns.")
#     st.stop()

# prediction_columns = {
#     "filename",
#     "label",
#     "predicted_label",
#     "spoof_probability",
# }
# if not prediction_columns.issubset(predictions.columns):
#     st.error(
#         "The prediction CSV needs filename, label, predicted_label, "
#         "and spoof_probability columns."
#     )
#     st.stop()

# # Match predictions to the actual labels by filename.
# data = predictions.merge(
#     features[["filename", "label"]].rename(columns={"label": "actual_label"}),
#     on="filename",
#     how="left",
#     validate="one_to_one",
# )

# # --------------------------------------------------
# # Dataset overview
# # --------------------------------------------------
# st.subheader("Dataset Overview")

# total_recordings = len(features)
# genuine_count = int((features["label"] == "genuine").sum())
# spoof_count = int((features["label"] == "spoof").sum())

# c1, c2, c3 = st.columns(3)
# c1.metric("Total Recordings", f"{total_recordings:,}")
# c2.metric("Genuine Recordings", f"{genuine_count:,}")
# c3.metric("Spoof Recordings", f"{spoof_count:,}")

# left, right = st.columns(2)

# with left:
#     st.markdown("#### Actual Dataset Distribution")
#     distribution = (
#         features["label"]
#         .value_counts()
#         .reindex(["genuine", "spoof"], fill_value=0)
#     )
#     st.bar_chart(distribution)

# with right:
#     st.markdown("#### Model Predictions Across All Recordings")
#     predicted_counts = (
#         data["predicted_label"]
#         .value_counts()
#         .reindex(["genuine", "spoof"], fill_value=0)
#     )
#     st.bar_chart(predicted_counts)

# st.caption(
#     "Predictions above cover the full available training dataset. "
#     "They are not independent test-set results."
# )

# st.divider()


# # --------------------------------------------------
# # Validation metrics
# # --------------------------------------------------
# st.subheader("Model Validation Performance")

# st.write(
#     "Random Forest classifier using 13 averaged MFCC features. "
#     "Metrics are loaded from the latest training run."
# )

# if metrics is not None:
#     # Main metrics
#     m1, m2, m3 = st.columns(3)
#     m1.metric(
#         "Validation Accuracy",
#         f"{metrics['accuracy'] * 100:.2f}%"
#     )
#     m2.metric(
#         "Validation ROC-AUC",
#         f"{metrics['roc_auc']:.4f}"
#     )
#     m3.metric(
#         "Approximate EER",
#         f"{metrics['eer_approximate'] * 100:.2f}%"
#     )

#     # Error rates
#     m4, m5, m6 = st.columns(3)
#     m4.metric(
#         "FAR",
#         f"{metrics['far_spoof_accepted_as_genuine'] * 100:.2f}%"
#     )
#     m5.metric(
#         "FRR",
#         f"{metrics['frr_genuine_rejected_as_spoof'] * 100:.2f}%"
#     )
#     m6.metric(
#         "Decision Threshold",
#         f"{metrics['threshold_for_far_frr']:.2f}"
#     )

#     st.caption(
#         "FAR = spoof recordings incorrectly accepted as genuine. "
#         "FRR = genuine recordings incorrectly rejected as spoof. "
#         "EER is an approximate equal error rate."
#     )

#     st.markdown("#### Confusion Matrix")
#     st.caption("Rows = actual labels; columns = predicted labels.")

#     confusion = pd.DataFrame(
#         metrics["confusion_matrix"],
#         index=["Actual Genuine", "Actual Spoof"],
#         columns=["Predicted Genuine", "Predicted Spoof"],
#     )
#     st.dataframe(confusion, use_container_width=True)

#     st.markdown("#### Precision, Recall and F1-Score")

#     report = metrics["classification_report"]

#     classification_report_data = {
#         "Metric": ["Precision", "Recall", "F1-score"],
#         "Genuine": [
#             f"{report['genuine']['precision'] * 100:.2f}%",
#             f"{report['genuine']['recall'] * 100:.2f}%",
#             f"{report['genuine']['f1-score'] * 100:.2f}%",
#         ],
#         "Spoof": [
#             f"{report['spoof']['precision'] * 100:.2f}%",
#             f"{report['spoof']['recall'] * 100:.2f}%",
#             f"{report['spoof']['f1-score'] * 100:.2f}%",
#         ],
#     }

#     st.dataframe(
#         pd.DataFrame(classification_report_data),
#         hide_index=True,
#         use_container_width=True,
#     )

#     st.markdown("#### Validation Details")
#     st.write(
#         f"Training samples: {metrics['training_samples']:,}  \n"
#         f"Validation samples: {metrics['validation_samples']:,}"
#     )

#     st.caption(
#         f"Evaluation: {metrics['evaluation_type']}"
#     )

#     st.warning(
#         "These results are from a stratified holdout of the available "
#         "training dataset, not the official ASVspoof evaluation set "
#         "or a real-world deployment."
#     )

# else:
#     st.info(
#         "Validation metrics are not available yet. "
#         "Run scripts/train_model.py, then refresh this dashboard."
#     )

# st.divider()

# # -------------------------------------------------
# # -
# # Prediction explorer
# # --------------------------------------------------
# st.subheader("Recording Prediction Explorer")

# search = st.text_input("Search by filename", placeholder="e.g. T_1000001.wav")
# selected_label = st.selectbox(
#     "Filter by predicted class",
#     ["All", "genuine", "spoof"],
# )

# filtered = data.copy()

# if search.strip():
#     filtered = filtered[
#         filtered["filename"].str.contains(search.strip(), case=False, na=False)
#     ]

# if selected_label != "All":
#     filtered = filtered[filtered["predicted_label"] == selected_label]

# display_columns = [
#     "filename",
#     "actual_label",
#     "predicted_label",
#     "spoof_probability",
# ]
# st.dataframe(
#     filtered[display_columns],
#     use_container_width=True,
#     hide_index=True,
# )

# st.caption(f"Showing {len(filtered):,} of {len(data):,} recordings.")

# csv_bytes = filtered[display_columns].to_csv(index=False).encode("utf-8")
# st.download_button(
#     "Download filtered predictions (CSV)",
#     data=csv_bytes,
#     file_name="voiceguard_predictions.csv",
#     mime="text/csv",
# )

# st.divider()
# st.caption(
#     "VoiceGuard is a research prototype. It currently demonstrates "
#     "voice anti-spoofing using MFCC features and a Random Forest model; "
#     "it does not yet implement face authentication or the complete "
#     "risk-adaptive authentication pipeline."
# )
