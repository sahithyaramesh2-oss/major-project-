"""Generate a standalone browser dashboard from measured result files."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


PLOTS = [
    ("svm_rbf_performance_comparison.png", "RBF SVM performance comparison"),
    ("svm_rbf_lbp_confusion_matrix.png", "LBP confusion matrix"),
    ("svm_rbf_lbp_roc.png", "LBP ROC curve"),
    ("svm_rbf_lbp_precision_recall.png", "LBP precision-recall curve"),
    ("svm_rbf_lbp_score_distribution.png", "LBP genuine vs spoof scores"),
    ("svm_rbf_lbp_far_frr.png", "LBP FAR vs FRR"),
    ("svm_rbf_hog_confusion_matrix.png", "HOG confusion matrix"),
    ("svm_rbf_hog_roc.png", "HOG ROC curve"),
    ("svm_rbf_hog_precision_recall.png", "HOG precision-recall curve"),
    ("svm_rbf_hog_score_distribution.png", "HOG genuine vs spoof scores"),
    ("svm_rbf_hog_far_frr.png", "HOG FAR vs FRR"),
    ("svm_rbf_combined_confusion_matrix.png", "Combined-feature confusion matrix"),
    ("svm_rbf_combined_roc.png", "Combined-feature ROC curve"),
    ("svm_rbf_combined_precision_recall.png", "Combined-feature precision-recall curve"),
    ("svm_rbf_combined_score_distribution.png", "Combined-feature genuine vs spoof scores"),
    ("svm_rbf_combined_far_frr.png", "Combined-feature FAR vs FRR"),
]
LEGACY_PLOTS = [
    ("accuracy_comparison.png", "Accuracy"),
    ("acer_comparison.png", "ACER"),
    ("eer_comparison.png", "Equal error rate"),
    ("roc_curves.png", "ROC curves"),
    ("confusion_matrices.png", "Confusion matrices"),
    ("apcer_comparison.png", "APCER"),
    ("bpcer_comparison.png", "BPCER"),
    ("class_distribution.png", "Class distribution"),
    ("videos_per_subject.png", "Videos per subject"),
    ("sample_frames.png", "Sampled frames"),
]


def _percent(value: float, deviation: float | None = None) -> str:
    if pd.isna(value):
        return "N/A"
    text = f"{value * 100:.2f}%"
    if deviation is not None and pd.notna(deviation):
        text += f" ± {deviation * 100:.2f}%"
    return text


def _subject_sort_key(subject_id: str) -> tuple[int, str]:
    return (0, f"{int(subject_id):08d}") if subject_id.isdigit() else (1, subject_id)


def generate_dashboard(inventory: pd.DataFrame, results_dir: str | Path) -> Path:
    results = Path(results_dir)
    svm_results = results / "svm_rbf"
    comparison_path = svm_results / "csv" / "model_comparison.csv"
    has_svm_results = comparison_path.is_file()
    if not has_svm_results:
        comparison_path = results / "csv" / "model_comparison.csv"
    if not comparison_path.is_file():
        raise FileNotFoundError("Run --phase train or --phase all before generating the results dashboard.")
    comparison = pd.read_csv(comparison_path)
    if comparison.empty:
        raise ValueError(f"No measured model results in {comparison_path}")

    smoke_test = False
    metadata_path = svm_results / "experiment_metadata.json"
    if has_svm_results and metadata_path.is_file():
        experiment_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        smoke_test = bool(experiment_metadata.get("smoke_test", False))
    notice_title = (
        "SMOKE TEST — DO NOT REPORT AS SCIENTIFIC EVALUATION."
        if smoke_test
        else "Preliminary evaluation, not the official benchmark."
    )

    best = comparison.sort_values("acer_mean", ascending=True).iloc[0]
    genuine_count = int((inventory["label"] == 0).sum())
    attack_count = int((inventory["label"] == 1).sum())
    subjects = int(inventory["subject_id"].nunique())
    category_counts = (
        inventory.loc[inventory["label"] == 1, "attack_category"]
        .value_counts().sort_index()
    )

    if has_svm_results:
        table_headers = (
            "<th>Dataset / features</th><th>Samples (live / attack)</th>"
            "<th>Validation accuracy</th><th>Test accuracy</th><th>Precision</th>"
            "<th>Recall</th><th>F1</th><th>FAR</th><th>FRR</th><th>EER</th>"
            "<th>ROC-AUC</th><th>APCER</th><th>BPCER</th><th>ACER</th>"
            "<th>Best C</th><th>Best gamma</th><th>Decision threshold</th>"
        )
        rows = []
        for row in comparison.itertuples(index=False):
            rows.append(
                "<tr>"
                f"<th scope=\"row\">{html.escape(str(row.dataset_name))} / {html.escape(str(row.feature_set).upper())}</th>"
                f"<td>{int(row.total_samples):,} ({int(row.genuine_samples):,} / {int(row.spoof_samples):,})</td>"
                f"<td>{_percent(row.validation_accuracy_mean, row.validation_accuracy_std)}</td>"
                f"<td>{_percent(row.accuracy_mean, row.accuracy_std)}</td>"
                f"<td>{_percent(row.precision_mean, row.precision_std)}</td>"
                f"<td>{_percent(row.recall_mean, row.recall_std)}</td>"
                f"<td>{_percent(row.f1_mean, row.f1_std)}</td>"
                f"<td>{_percent(row.far_mean, row.far_std)}</td>"
                f"<td>{_percent(row.frr_mean, row.frr_std)}</td>"
                f"<td>{_percent(row.eer_mean, row.eer_std)}</td>"
                f"<td>{row.roc_auc_mean:.4f} ± {row.roc_auc_std:.4f}</td>"
                f"<td>{_percent(row.apcer_mean, row.apcer_std)}</td>"
                f"<td>{_percent(row.bpcer_mean, row.bpcer_std)}</td>"
                f"<td>{_percent(row.acer_mean, row.acer_std)}</td>"
                f"<td>{html.escape(str(row.best_c))}</td>"
                f"<td>{html.escape(str(row.best_gamma))}</td>"
                f"<td>{_percent(row.threshold_mean, row.threshold_std)}</td>"
                "</tr>"
            )
        plot_dir = svm_results / "plots"
        plot_relative_dir = "svm_rbf/plots"
    else:
        table_headers = (
            "<th>Model / features</th><th>Accuracy</th><th>APCER</th>"
            "<th>BPCER</th><th>ACER</th><th>EER</th><th>ROC-AUC</th>"
        )
        rows = []
        for row in comparison.itertuples(index=False):
            rows.append(
                "<tr>"
                f"<th scope=\"row\">{html.escape(str(row.model))}</th>"
                f"<td>{_percent(row.accuracy_mean, row.accuracy_std)}</td>"
                f"<td>{_percent(row.apcer_mean, row.apcer_std)}</td>"
                f"<td>{_percent(row.bpcer_mean, row.bpcer_std)}</td>"
                f"<td>{_percent(row.acer_mean, row.acer_std)}</td>"
                f"<td>{_percent(row.eer_mean, row.eer_std)}</td>"
                f"<td>{row.roc_auc_mean:.4f} ± {row.roc_auc_std:.4f}</td>"
                "</tr>"
            )
        plot_dir = results / "plots"
        plot_relative_dir = "plots"

    if has_svm_results:
        best_label = f"{best.dataset_name} / {str(best.feature_set).upper()}"
        subtitle = (
            "Video-level evaluation of the same scaled RBF SVM across the available "
            "face-feature representations. Each plotted prediction comes from a held-out test subject."
        )
        notice_text = (
            "Subject-disjoint local evaluation; not an official benchmark claim. "
            f"Available subject IDs: {', '.join(sorted(inventory['subject_id'].astype(str).unique(), key=_subject_sort_key))}. "
            "The published protocol requires dataset partitions not established by this local run."
        )
        protocol_text = (
            "Each available subject is held out once for testing; a different subject is used for "
            "validation and all other subjects are used for training. Grouped cross-validation, "
            "scaling, and hyperparameter selection use training subjects only. The validation "
            "subject selects the fixed operating threshold. Class 0 is genuine/live; class 1 is "
            "spoof/attack. FAR equals APCER and FRR equals BPCER under these definitions."
        )
        summary_links = (
            '<a href="svm_rbf/csv/model_comparison.csv">SVM comparison CSV</a>'
            '<a href="svm_rbf/csv/fold_metrics.csv">Fold metrics CSV</a>'
            '<a href="svm_rbf/csv/video_predictions.csv">Test predictions CSV</a>'
            '<a href="svm_rbf/csv/split_manifest.csv">Split manifest CSV</a>'
            '<a href="svm_rbf/experiment_metadata.json">Experiment metadata</a>'
            '<a href="svm_rbf/models/model_manifest.json">Saved model manifest</a>'
            '<a href="svm_rbf/client_dataset_status.json">Auxiliary client split status</a>'
        )
        client_status_path = svm_results / "client_dataset_status.json"
        if client_status_path.is_file():
            client_status = json.loads(client_status_path.read_text(encoding="utf-8"))
            reason = client_status.get("reason_not_evaluated")
            auxiliary_note = (
                f"Auxiliary client split not evaluated: {html.escape(reason)}"
                if reason
                else "Auxiliary client split is eligible for a separate experiment but is not part of this run."
            )
        else:
            auxiliary_note = "Auxiliary client split status has not been generated."
        table_help = (
            "Entries show mean ± standard deviation across held-out-subject test folds. "
            "The displayed C and gamma are the most frequently selected training-CV values; "
            "all fold selections and validation thresholds are in the downloadable fold metrics."
        )
    else:
        best_label = str(best.model)
        subtitle = (
            "A video-level comparison of lightweight LBP/HOG features with SVM and Random Forest. "
            "Results below are generated from the local dataset run."
        )
        notice_text = (
            "Preliminary subject-disjoint evaluation, not the official benchmark. "
            f"Available subject IDs: {', '.join(sorted(inventory['subject_id'].astype(str).unique(), key=_subject_sort_key))}."
        )
        protocol_text = (
            "Each fold holds out a distinct subject for test and a second subject for validation; "
            "all remaining subjects are used for training. Test decisions use the ACER-minimizing "
            "validation threshold. Higher scores indicate spoof (class 1)."
        )
        summary_links = (
            '<a href="csv/model_comparison.csv">Model comparison CSV</a>'
            '<a href="csv/fold_metrics.csv">Fold metrics CSV</a>'
            '<a href="csv/video_predictions.csv">Video predictions CSV</a>'
            '<a href="csv/attack_category_apcer.csv">Attack-category APCER CSV</a>'
        )
        auxiliary_note = ""
        table_help = (
            "Entries show mean ± standard deviation across held-out-subject folds. "
            "Rates are percentages; ROC-AUC is unit scale."
        )

    category_rows = "\n".join(
        f"<tr><th scope=\"row\">{html.escape(str(category).replace('_', ' ').title())}</th>"
        f"<td>{int(count):,}</td></tr>"
        for category, count in category_counts.items()
    )
    plot_list = PLOTS if has_svm_results else LEGACY_PLOTS
    plot_cards = "\n".join(
        f'<figure class="chart-card"><figcaption>{html.escape(title)}</figcaption>'
        f'<a href="{plot_relative_dir}/{filename}" target="_blank" rel="noopener">'
        f'<img src="{plot_relative_dir}/{filename}" alt="{html.escape(title)} plot" loading="lazy"></a></figure>'
        for filename, title in plot_list
        if (plot_dir / filename).is_file()
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Measured preliminary ROSE-Youtu face presentation attack detection experiment results">
  <title>ROSE-Youtu | Experiment Results</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1020; --panel:#121a2d; --line:#26334d; --muted:#9baac1; --text:#edf3ff; --accent:#67d5b5; --blue:#79aaff; --warning:#f5ca75; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(ellipse at 75% -10%,#21365a 0,transparent 38%),var(--bg); color:var(--text); font:15px/1.55 Inter,Segoe UI,Arial,sans-serif; }}
    header {{ padding:40px max(24px,calc((100vw - 1200px)/2)) 28px; border-bottom:1px solid var(--line); }}
    .eyebrow {{ color:var(--accent); font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
    h1 {{ max-width:900px; margin:9px 0; font-size:clamp(2rem,4vw,3rem); line-height:1.12; }}
    .subtitle {{ color:var(--muted); max-width:850px; }}
    main {{ max-width:1200px; margin:auto; padding:24px; }}
    .notice {{ border:1px solid #806533; background:#322818; border-radius:12px; padding:15px 18px; color:#f8e6bf; margin-bottom:22px; }}
    .notice strong {{ color:var(--warning); }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:18px 0 28px; }}
    .stat,.panel {{ background:linear-gradient(145deg,#151f34,#111827); border:1px solid var(--line); border-radius:14px; }}
    .stat {{ padding:18px; }}
    .stat span {{ display:block; color:var(--muted); font-size:.83rem; }}
    .stat strong {{ display:block; font-size:1.7rem; margin-top:5px; }}
    section {{ margin:30px 0; }}
    h2 {{ font-size:1.35rem; margin:0 0 12px; }}
    .panel {{ padding:20px; overflow-x:auto; }}
    .best {{ border-left:3px solid var(--accent); }}
    .best-name {{ color:var(--accent); font-weight:700; font-size:1.25rem; }}
    .best p {{ color:var(--muted); margin:7px 0 0; }}
    table {{ width:100%; border-collapse:collapse; min-width:850px; }}
    th,td {{ text-align:left; padding:11px 12px; border-bottom:1px solid var(--line); white-space:nowrap; }}
    thead th {{ color:#b9c9e2; font-size:.8rem; text-transform:uppercase; letter-spacing:.04em; }}
    tbody th {{ font-weight:650; }}
    tbody tr:first-child {{ background:#172a31; }}
    .help {{ color:var(--muted); font-size:.88rem; }}
    .charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr)); gap:15px; }}
    .chart-card {{ margin:0; padding:14px; background:var(--panel); border:1px solid var(--line); border-radius:13px; }}
    .chart-card figcaption {{ font-weight:650; margin-bottom:9px; }}
    .chart-card img {{ display:block; width:100%; height:auto; border-radius:7px; background:white; }}
    .lower {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr)); gap:16px; }}
    .links a {{ display:inline-block; margin:4px 10px 4px 0; color:var(--blue); }}
    footer {{ border-top:1px solid var(--line); padding:22px 24px; color:var(--muted); text-align:center; font-size:.85rem; }}
    @media(max-width:600px) {{ main {{ padding:16px; }} header {{ padding:28px 18px 20px; }} .panel {{ padding:14px; }} }}
  </style>
</head>
<body>
<header>
  <div class="eyebrow">ROSE-Youtu · Face presentation attack detection</div>
  <h1>Experiment results</h1>
  <div class="subtitle">{subtitle}</div>
</header>
<main>
  <div class="notice"><strong>{notice_title}</strong> {html.escape(notice_text)}</div>
  <div class="stats">
    <div class="stat"><span>Available subjects</span><strong>{subjects}</strong></div>
    <div class="stat"><span>Videos inspected</span><strong>{len(inventory):,}</strong></div>
    <div class="stat"><span>Genuine videos</span><strong>{genuine_count:,}</strong></div>
    <div class="stat"><span>Attack videos</span><strong>{attack_count:,}</strong></div>
    <div class="stat"><span>Best mean ACER</span><strong>{best.acer_mean * 100:.2f}%</strong></div>
  </div>
  <section class="panel best" aria-label="Best model by mean ACER">
    <div class="eyebrow">Lowest mean fold ACER</div>
    <div class="best-name">{html.escape(best_label)}</div>
    <p>Accuracy {_percent(best.accuracy_mean)} · APCER {_percent(best.apcer_mean)} · BPCER {_percent(best.bpcer_mean)} · EER {_percent(best.eer_mean)} · ROC-AUC {best.roc_auc_mean:.4f}</p>
    <p>Ranked by validation-threshold fixed operating ACER across held-out subjects. EER is calculated retrospectively from continuous video scores.</p>
  </section>
  <section>
    <h2>Model comparison</h2>
    <div class="panel">
      <table>
        <thead><tr>{table_headers}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="help">{table_help}</p>
    </div>
  </section>
  <section class="lower">
    <div>
      <h2>Observed attack categories</h2>
      <div class="panel"><table><thead><tr><th>Category</th><th>Videos</th></tr></thead><tbody>{category_rows}</tbody></table></div>
    </div>
    <div>
      <h2>Protocol & interpretation</h2>
      <div class="panel">
        <p>{protocol_text} All metrics and plots are based on independent held-out test predictions.</p>
        <p>These local results do not establish official benchmark performance or real-world security.</p>
        <p>{auxiliary_note}</p>
        <div class="links">{summary_links}</div>
      </div>
    </div>
  </section>
  <section>
    <h2>Figures</h2>
    <div class="charts">{plot_cards}</div>
  </section>
  <section class="panel">
    <h2>Downloadable data</h2>
    <div class="links">{summary_links}<a href="dataset_inventory.csv">Dataset inventory CSV</a><a href="../datasets/metadata/video_labels.csv">Verified video labels CSV</a></div>
  </section>
</main>
<footer>ROSE-Youtu face PAD · Local experiment outputs · No identity recognition or voice authentication</footer>
</body>
</html>
"""
    target = results / "dashboard.html"
    target.write_text(page, encoding="utf-8")
    return target
