# ROSE-Youtu Face Liveness Detection

CPU-friendly face presentation attack detection (PAD) using the ROSE-Youtu dataset. The active training pipeline uses a common RBF-kernel SVM for the existing sampled visual texture/gradient features. This project does **not** recognize identity or authenticate voice.

## Dataset and scientific scope

The official dataset page documents the filename format, attack codes, academic-use terms, and protocols: <https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/>. The original source videos belong under `datasets/original/<subject-id>/`. The project discovers folders dynamically and never changes the source files. `datasets/client/client/rgb/` is an auxiliary processed split from the same face dataset, not a separate modality. Its adaptation list currently contains only bona fide labels and its local test images do not cover the full annotation list, so it is recorded as unavailable for an independent SVM experiment rather than mixed with original-video samples.

The refreshed local inventory contains 3,137 face-liveness videos from 18 subjects (797 genuine and 2,340 attacks); all passed the inventory decode check, with no duplicate video IDs. The source remains a partial ROSE-Youtu copy, not the complete official benchmark. No voice-liveness or separate video-liveness dataset is present locally; those modalities remain pending rather than receiving invented metrics.

The local snapshot may be incomplete. The official intra-dataset protocol requires training subjects 2, 3, 4, 5, 6, 7, 9, 10, 11, and 12 and a held-out test partition. The inspection report records the subjects currently present; the pipeline reports a **preliminary subject-disjoint experiment**, not an official benchmark result, unless the complete authorized split is available.

Official filename labels: the first field `G` means genuine (label 0); the documented `Ps`, `Pq`, `Vl`, `Vm`, `Mc`, `Mf`, `Mu`, and `Ml` codes represent attacks (label 1). Unsupported names fail parsing; labels are never guessed.

## Structure

```text
datasets/
  original/                 Original videos, unmodified
  client/                   Auxiliary incomplete ROSE-Youtu RGB split
  metadata/video_labels.csv Parsed verified labels
  processed/frames/         Sampled grayscale frames
  processed/features/       Cached compressed feature matrices
src/rose_liveness/          Dataset, video, feature, metrics, SVM experiment, reports
tests/                      Parser, metrics, and SVM pipeline tests
results/
  dataset_inventory.csv
  dataset_inspection.md
  csv/                      Class counts, splits, frame status, predictions, metrics
  plots/                    PNG figures
  tables/model_comparison.md
  reports/                  Evaluation protocol and paper-ready content
  svm_rbf/                  Tuned SVM metrics, split manifest, plots, and models
  dashboard.html            Standalone browser results dashboard
```

## Technology

Python 3.10+, OpenCV, NumPy, Pandas, scikit-image, scikit-learn, SciPy, Matplotlib, Joblib, tqdm, pytest.

## Setup (Windows PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project also works with a compatible installed Python, using that interpreter in place of `py -3.10`.

## Run

From the project root:

```powershell
python main.py --phase inspect
python main.py --phase preprocess
python main.py --phase features
python main.py --phase train
python main.py --phase plots
python main.py --phase dashboard
python main.py --phase all
python main.py --phase all --smoke-test
```

`inspect` discovers MP4 files, reads media properties, checks the first frame for decodability, writes the full inventory and verified label CSV, and reports class counts. Existing inventory files are refreshed when their video paths no longer match the dataset folders. `preprocess` samples five uniformly spaced frames per video, crops a detected face with a margin when a supported OpenCV Haar API is available, and otherwise retains/logs a resized full-frame fallback. `features` extracts global uniform LBP and HOG descriptors and caches them under `datasets/processed/features/`. `train` and `evaluate` run tuned RBF SVMs separately on LBP, HOG, and combined features. `plots` regenerates test-set figures and the dashboard. `dashboard` regenerates the standalone results page without rerunning training. `all` performs the full available-data workflow.

To view the dashboard, generate it with `python main.py --phase dashboard`, then either open `results/dashboard.html` in a browser or serve the project root and visit `http://localhost:8000/results/dashboard.html`:

```powershell
python -m http.server 8000
```

The smoke test samples real videos from four subjects so grouped CV, validation, and testing can use disjoint subject groups. It uses a separate feature cache so it does not replace the full-dataset cache. Its output is a pipeline check only and must not be presented as a scientific evaluation. Cached frames and features allow later commands to resume without repeating preprocessing. Fold models are saved as joblib-compatible `.pkl` files under `results/svm_rbf/models/folds/`.

The dataset inspection and frame-extraction summary record the current local subject coverage, decode failures, and whether face crops or full-frame fallbacks were used. Official benchmark coverage and preprocessing support must be verified from these generated reports; do not infer them from prior runs.

## Evaluation protocol and metrics

The evaluation unit is a video: sampled-frame descriptors are averaged into one video feature vector. Five outer `StratifiedGroupKFold` folds hold each subject out for testing once. Each outer training partition is split again by subject into training and validation groups; `GridSearchCV` tunes C and gamma using three-fold `StratifiedGroupKFold` grouped by training subject, with four bounded parallel search workers. The scaler is inside the SVM `Pipeline`, so every CV scaler is fitted only on its training fold and the final fold estimator is fitted only on outer-training subjects. The operating threshold minimizes validation ACER and is fixed before scoring each held-out test fold. SVM scores are class-1 spoof probabilities. EER is a retrospective test-score threshold sweep, distinct from fixed-threshold test performance. FAR is reported as APCER (attacks accepted as genuine); FRR is BPCER (genuine samples rejected as attacks).

The output includes accuracy, precision, recall, F1, FAR, FRR, EER, ROC-AUC, APCER, BPCER, ACER, and confusion matrices. Figures include ROC, precision-recall, genuine-versus-spoof score distribution, FAR-versus-FRR, and a feature comparison chart. The dashboard compares the three feature representations using fold means and standard deviations, reports the most frequently selected C/gamma, and links to every fold's actual choices and thresholds.

### Measured local experiment

The completed run evaluated 3,137 videos (797 genuine, 2,340 attacks) in five subject-disjoint outer folds for each feature representation. Values below are means over the five independent test folds; C/gamma are the most frequently selected training-only CV values, and the threshold is the mean of the per-fold validation-selected thresholds. FAR equals APCER and FRR equals BPCER under the authentication definitions above.

| Features | Validation accuracy | Test accuracy | Precision | Recall | F1 | FAR / APCER | FRR / BPCER | EER | ROC-AUC | ACER | Modal C / gamma | Mean threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| LBP | 0.7678 | 0.7811 | 0.9100 | 0.7863 | 0.8421 | 0.2137 | 0.2441 | 0.2233 | 0.8486 | 0.2289 | 1 / scale | 0.7354 |
| HOG | 0.9190 | 0.8953 | 0.9631 | 0.8945 | 0.9270 | 0.1055 | 0.1211 | 0.1072 | 0.9592 | 0.1133 | 1 / scale | 0.8625 |
| Combined LBP+HOG | 0.9146 | 0.8939 | 0.9716 | 0.8840 | 0.9252 | 0.1160 | 0.0963 | 0.1007 | 0.9634 | 0.1062 | 1 / scale | 0.8960 |

These are preliminary results on the locally available partial dataset, not official ROSE-Youtu benchmark results. Per-fold results and thresholds are in `results/svm_rbf/csv/fold_metrics.csv` and `results/svm_rbf/csv/split_manifest.csv`; the fold-mean comparison is in `results/svm_rbf/csv/model_comparison.csv`. No voice or independent video dataset was available for evaluation.

The tuned experiment is written under `results/svm_rbf/`; the prior comparison files and trained models under `results/csv/` and `results/models/` are retained. The `results/dashboard.html` page shows the tuned SVM experiment when its results are available.

## Reproducibility and limitations

Random seeds are fixed. Original videos remain untouched. Subject/video IDs, fold membership, frame fallback status, model settings, and measured predictions are recorded. The lightweight Haar detector, handcrafted descriptors, partial subject set, and small preliminary evaluation limit generalization. Official results require the complete authorized data and published split. Dataset access is governed by the academic release agreement.

Run tests with:

```powershell
pytest
```