
# Voice Anti-Spoofing Module

## 1. Overview

This module detects whether an audio recording is genuine or spoofed using machine learning classification.

It is developed as the voice anti-spoofing component of a multimodal biometric authentication research project. Its purpose is to identify potentially spoofed audio recordings and provide classification results for further authentication processing.

The final selected model is a hyperparameter-tuned Support Vector Machine (SVM) with an RBF kernel.

## 2. Dataset

**Dataset:** ASVspoof 2017 Version 2

The dataset contains genuine and spoofed audio recordings.

| Dataset Split | Genuine | Spoof | Total |
|---|---:|---:|---:|
| Training | 1,507 | 1,507 | 3,014 |
| Development | 760 | 950 | 1,710 |

The raw audio dataset is not included in this repository and must be obtained separately.

Expected dataset structure:

```text
voice_antispoofing/
└── dataset/
    ├── ASVspoof2017_V2_train/
    ├── ASVspoof2017_V2_dev/
    └── protocol_V2/
        └── ASVspoof2017_V2_dev.trl.txt
```

## 3. Feature Extraction

Audio recordings are converted to mono before feature extraction.

The following process is applied:

1. Load audio recordings using Librosa.
2. Extract 13 Mel-Frequency Cepstral Coefficients (MFCCs).
3. Calculate the mean of each MFCC coefficient across audio frames.
4. Generate a 13-dimensional feature vector for each recording.

These feature vectors are used as input to the machine learning classifiers.

## 4. Algorithms Implemented

Three classification approaches were implemented and evaluated:

1. **Random Forest Classifier:** Baseline machine learning model.
2. **Support Vector Machine (SVM):** SVM classifier with an RBF kernel.
3. **Hyperparameter-Tuned SVM:** SVM with an RBF kernel and optimized hyperparameters.

The hyperparameter-tuned SVM was selected as the final model.

## 5. Final Model Configuration

| Parameter | Configuration |
|---|---|
| Algorithm | Support Vector Machine |
| Kernel | RBF (Radial Basis Function) |
| Feature Scaling | StandardScaler |
| C | 1 |
| Gamma | 1 |
| Class Weight | Balanced |
| Probability Estimation | Disabled |
| Hyperparameter Tuning | GridSearchCV |
| Cross-Validation | 5-fold StratifiedKFold |
| Selection Metric | ROC-AUC |

### Hyperparameter Tuning

GridSearchCV was used to find the optimal SVM configuration.

Search space:

- C: 0.1, 1, 10, 100
- Gamma: scale, 0.001, 0.01, 0.1, 1
- Kernel: RBF

The best configuration was selected using five-fold stratified cross-validation with ROC-AUC as the scoring metric.

**Best cross-validation ROC-AUC:** 0.9998

## 6. Training and Validation Results

The training dataset was divided into:

- 80% training data: 2,411 recordings.
- 20% validation data: 603 recordings.

The tuned SVM was selected through cross-validation using ROC-AUC as the selection metric. Its performance was subsequently evaluated on the stratified holdout validation set and compared with the baseline models.

| Metric | Random Forest | Initial SVM | Tuned SVM |
|---|---:|---:|---:|
| Accuracy | 98.01% | 99.00% | 99.50% |
| ROC-AUC | 0.9979 | 0.9997 | 0.9989 |
| FAR | 0.66% | 1.00% | 0.33% |
| FRR | 3.31% | 0.99% | 0.66% |
| Approx. EER | 2.16% | 1.16% | 0.17% |

The tuned SVM achieved 99.50% validation accuracy and was retained as the final model.

## 7. Development Dataset Evaluation

The final tuned SVM was evaluated on the ASVspoof 2017 V2 development dataset containing 1,710 recordings.

An operating threshold of approximately 0.077554 was selected using Equal Error Rate (EER)-based threshold analysis on the development dataset.

### Results

| Metric | Result |
|---|---:|
| Accuracy | 87.37% |
| ROC-AUC | 0.9361 |
| FAR | 12.63% |
| FRR | 12.63% |
| Approx. EER | 12.63% |

The threshold is applied to the SVM decision scores during development evaluation.

**Important:** The operating threshold was calibrated using the development dataset itself. Therefore, these results should be reported as development-set evaluation results, not as results from an untouched independent test dataset.

## 8. Folder Structure

```text
voice_antispoofing/
│
├── app.py
│
├── README.md
│
├── models/
│   ├── voice_antispoof_model.joblib
│   ├── voice_antispoof_svm_model.joblib
│   └── voice_antispoof_svm_tuned_model.joblib
│
├── results/
│   ├── training_features.csv
│   ├── validation_metrics.json
│   ├── svm_validation_metrics.json
│   ├── svm_tuned_validation_metrics.json
│   ├── svm_dev_metrics.json
│   ├── svm_dev_predictions.csv
│   ├── svm_grid_search_results.csv
│   └── svm_threshold_analysis.csv
│
└── scripts/
    ├── train_model.py
    ├── train_svm.py
    ├── train_svm_tuned.py
    ├── evaluate_svm_dev.py
    └── analyze_dev_thresholds.py
```

## 9. Dependencies

The module uses the following Python libraries:

- Python
- NumPy
- Pandas
- Librosa
- Scikit-learn
- Joblib
- SoundFile
- Streamlit

Install the dependencies using:

```bash
pip install numpy pandas librosa scikit-learn joblib soundfile streamlit
```

## 10. Execution

Run the commands from the main repository directory.

### Step 1: Train and tune the SVM

```bash
python voice_antispoofing/scripts/train_svm_tuned.py
```

This performs hyperparameter tuning, trains the selected SVM model, evaluates it on the holdout validation set and saves the trained model and metrics.

### Step 2: Evaluate on the development dataset

Ensure that the development audio files and protocol file are available in the expected dataset directories.

```bash
python voice_antispoofing/scripts/evaluate_svm_dev.py
```

This generates development predictions and evaluation metrics.

### Step 3: Launch the voice dashboard

```bash
streamlit run voice_antispoofing/app.py
```

The dashboard displays dataset information, model configuration, validation and development metrics, threshold analysis and individual development predictions.

## 11. Limitations and Scope

- The current module classifies audio recordings as genuine or spoofed.
- It does not independently perform speaker identity verification.
- It does not perform face recognition or face liveness detection.
- It does not implement multimodal face-voice fusion or the final combined authentication decision.
- The Streamlit dashboard is a separate demonstration interface for the voice module.

The voice anti-spoofing classifier is intended to serve as a supporting component in the larger face-voice biometric authentication research project.

## 12. Final Model

The final selected classifier is the hyperparameter-tuned SVM with an RBF kernel, using 13 MFCC features and StandardScaler preprocessing.

The trained model is saved at:

```text
models/voice_antispoof_svm_tuned_model.joblib
```

Development evaluation metrics and individual predictions are saved in the `results/` directory.
