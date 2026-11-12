# Face and Voice Liveness Detection System

A liveness-detection project for distinguishing genuine users from spoofing attacks. **The current implementation is a video-based face presentation-attack detection (PAD) pipeline only. Voice/audio liveness detection and face–voice fusion are not implemented.**

## Current Implementation

- Face-based genuine-versus-spoof classification using the ROSE-Youtu face liveness dataset.
- Video preprocessing with uniformly sampled frames and face cropping when the available OpenCV Haar detector finds a face; otherwise, the resized full frame is used and recorded as a fallback.
- Uniform Local Binary Pattern (LBP) and Histogram of Oriented Gradients (HOG) feature extraction, with separate LBP, HOG, and concatenated-feature experiments.
- A common scaled Support Vector Machine with an RBF kernel for the available compatible face-feature representations.
- Subject-disjoint training, validation, and testing, with grouped cross-validation and validation-only decision-threshold selection.
- Accuracy, precision, recall, F1-score, FAR, FRR, EER, ROC-AUC, APCER, BPCER, ACER, confusion matrices, and ROC-curve evaluation.
- Plot generation and a standalone HTML dashboard sourced from saved experiment results.

Voice liveness, audio features, multimodal fusion, and real-time webcam inference are not part of the current code.

## Dataset

The implementation uses **ROSE-Youtu Face Liveness Detection** videos. The dataset contains bona fide face recordings and presentation attacks such as printed-photo and display-replay attacks. See the [official dataset page](https://rose1.ntu.edu.sg/dataset/faceLivenessDetection/) for access conditions and the complete dataset description.

The locally inventoried snapshot contains **2,622 videos** from 15 subjects: **672 genuine/live** and **1,950 spoof/attack** videos. These are local inventory counts, not counts for the complete official release. The dataset files are not included in this repository; place authorized original videos under `datasets/original/<subject-id>/`. Labels are parsed from the documented ROSE-Youtu filenames rather than inferred.

Preprocessing samples up to five uniformly spaced frames per video, converts them to grayscale, and resizes them to 128 × 128 pixels. Detected faces are cropped with a margin; full-frame fallback is recorded when a face crop is unavailable. The feature pipeline extracts a uniform LBP histogram and a HOG descriptor from each frame, then averages frame descriptors to form one video-level feature vector.

## Model Architecture

```text
ROSE-Youtu video
  → sampled grayscale frames and face-crop/full-frame preprocessing
  → LBP or HOG feature extraction (or compatible concatenation)
  → video-level feature aggregation
  → StandardScaler
  → SVM (RBF kernel)
  → genuine/live or spoof/attack prediction
```

The classifier is `sklearn.svm.SVC` with `kernel="rbf"`, `class_weight="balanced"`, probability estimates, and a fixed random seed. `StandardScaler` is inside the scikit-learn `Pipeline`. Grid search selects `C` from `[0.1, 1, 10, 100]` and `gamma` from `["scale", 0.001, 0.01, 0.1]` using grouped cross-validation on training subjects only. The validation subject selects the operating threshold; the independent held-out test subject is used for reported test metrics.

This same RBF-SVM family is used for the available LBP, HOG, and combined face features. These are separate feature representations of the same video modality, not voice or other datasets. Separate models are required if future modalities have incompatible feature dimensions.

## Performance Evaluation

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| Precision | Correct positive predictions |
| Recall | Detection capability |
| F1-score | Balance between precision and recall |
| FAR | False Acceptance Rate |
| FRR | False Rejection Rate |
| EER | Equal Error Rate |
| ROC-AUC | Classification performance across score thresholds |
| APCER | Attack Presentation Classification Error Rate |
| BPCER | Bona Fide Presentation Classification Error Rate |
| ACER | Average Classification Error Rate |

No full-dataset independent-test performance values are published here. A small real-data smoke test is available to check the pipeline, but it is not a scientific evaluation and its scores must not be interpreted as benchmark results. Run the full experiment to generate measured fold metrics, CSV/JSON summaries, confusion matrices, ROC plots, and the dashboard under `results/`.

Class 0 is genuine/live and class 1 is spoof/attack. In this implementation, FAR equals APCER and FRR equals BPCER. EER is calculated by sweeping test scores; the operating threshold itself is selected using validation data only.

## Project Structure

```text
liveness_detection/
├── .gitignore
├── README.md
├── experiment_config.json
├── main.py
├── requirements.txt
├── src/
│   └── rose_liveness/
│       ├── cli.py
│       ├── dashboard.py
│       ├── dataset.py
│       ├── experiment.py
│       ├── features.py
│       ├── labels.py
│       ├── metrics.py
│       ├── plots.py
│       ├── reports.py
│       ├── svm_experiment.py
│       ├── svm_plots.py
│       ├── video.py
│       └── __init__.py
└── tests/
    ├── test_labels_metrics.py
    └── test_svm_experiment.py
```

Downloaded videos, extracted frames, cached features, trained models, and generated result files are intentionally excluded from Git. The source code creates the required runtime directories.

## Installation

From the `liveness_detection/` directory, using Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Python 3.10 or a compatible newer version is required. Obtain the dataset from its official source under its applicable terms; do not commit dataset videos.

## How to Run

From the `liveness_detection/` directory:

```powershell
# Inspect and verify local video filenames and labels
python main.py --phase inspect

# Run the end-to-end available-data workflow
python main.py --phase all

# Or run individual stages
python main.py --phase preprocess
python main.py --phase features
python main.py --phase train
python main.py --phase plots
python main.py --phase dashboard

# Run tests
pytest
```

The first full run samples frames and extracts features, which can take time and disk space. Later runs reuse valid cached frames and features. For a small pipeline check, use `python main.py --phase all --smoke-test`; smoke-test outputs are not scientific results. The dashboard is written to `results/dashboard.html` after training or dashboard generation. To serve it locally, run `python -m http.server 8000` and open `http://localhost:8000/results/dashboard.html`.

## Technologies Used

- Python
- OpenCV
- NumPy and Pandas
- scikit-image
- scikit-learn
- Matplotlib
- Joblib
- pytest

Streamlit is not used; the dashboard is a standalone HTML page.

## Future Improvements

- Add separately sourced and documented voice liveness datasets and audio feature pipelines.
- Evaluate additional face and video PAD datasets and attack types.
- Run cross-dataset and official-protocol evaluations when the complete authorized data and splits are available.
- Investigate multimodal face–voice fusion after compatible modality-specific models are implemented.
- Explore real-time inference and stronger face/attack detection methods.
