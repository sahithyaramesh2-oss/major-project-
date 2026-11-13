# Evaluation protocol

## Dataset coverage

The local original-video inventory contains 3,137 ROSE-Youtu face-liveness videos from 18 subjects: 797 genuine and 2,340 attacks. All inventoried videos passed the current decode check, and no duplicate file paths or video IDs were found. This is a partial local copy of the official dataset; the complete published test partition is not verified, so results are preliminary and are not claimed as the official benchmark.

The original dataset is video-based face presentation-attack detection. No independent voice-liveness or additional video-liveness dataset is present in the workspace. The auxiliary `datasets/client/` RGB split is not evaluated: its adaptation annotation contains only bona fide labels, and its local test images do not cover the full annotation list.

## Split and evaluation unit

The video is the evaluation unit. Five outer `StratifiedGroupKFold` folds hold every subject out for testing once. Within each outer training partition, a second grouped split creates disjoint training and validation subject groups. Every model's hyperparameter search uses training data only, with three-fold `StratifiedGroupKFold` grouped by training subject. Test subjects are not used for parameter selection, scaling, or threshold selection.

Five uniformly spaced frames are sampled per video, resized to 128 x 128 grayscale, and cropped to a detected face when supported; logged full-frame fallback is used otherwise. Uniform LBP and HOG descriptors are extracted per frame, and frame features are averaged to form one video feature vector. LBP, HOG, and their concatenation are evaluated separately; incompatible feature vectors are never mixed.

## Classifier and tuning

All evaluated feature sets use the same `StandardScaler` + RBF-kernel `SVC` pipeline with class weighting, probability scores, and random seed 42. `GridSearchCV` searches `C` in `[0.1, 1, 10, 100]` and `gamma` in `["scale", 0.001, 0.01, 0.1]`, selecting by balanced accuracy with four parallel workers. The scaler is fitted inside each training fold. An operating threshold minimizing validation ACER is fixed before scoring the held-out test fold.

## Metrics

Class 0 is genuine/live (bona fide); class 1 is spoof/attack. A larger score means more likely spoof. APCER/FAR is the fraction of attacks accepted as genuine; BPCER/FRR is the fraction of genuine samples rejected as attacks; ACER is their mean. Accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and operating error rates use the fixed validation-derived threshold where applicable. EER is calculated retrospectively by sweeping held-out test score thresholds and is not the operating threshold.

Fold metrics are summarized as mean and standard deviation across held-out subjects. The saved split manifest, out-of-fold predictions, comparison CSV, per-feature metrics JSON, model manifest, and figures provide the measured results and audit trail.
