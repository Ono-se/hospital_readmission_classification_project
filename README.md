# Predicting 30-Day Hospital Readmission

## Business Problem

Unplanned 30-day readmissions are costly and disruptive: they strain hospital capacity, drive penalties under value-based care programs, and often signal that a patient was discharged before they were ready or without adequate follow-up support. For a hospital or payer, the practical question isn't "can we predict readmission?" — it's **"which patients, at the point of discharge, should get extra attention from a case manager before they leave?"**

This project builds a model that flags patients at elevated risk of readmission within 30 days, using information available at discharge: demographics, diagnoses, prior utilization, and treatment details. The intended use is to support **triage, not diagnosis** — a positive flag should prompt a follow-up call, a medication review, or a closer discharge-planning conversation, not an automated clinical decision.

## Why This Is Framed as Cost-Asymmetric

A missed readmission (false negative) is more expensive — in both patient outcomes and downstream cost — than an unnecessary follow-up call (false positive). That asymmetry drives two modeling choices throughout this project:

- **Class weighting** (`class_weight="balanced"`) on every model, so the minority class isn't ignored in favor of overall accuracy.
- **Threshold selection by F1 on the positive class**, rather than defaulting to 0.5, so the final operating point reflects the actual cost trade-off rather than an arbitrary cutoff.

## Data

- **Source:** UCI "Diabetes 130-US hospitals" dataset — 101,766 patient encounters, 50 variables (demographics, admission details, diagnoses, medications, and utilization history).
- **Target:** `readmitted` was collapsed into a binary label — **1** if the patient was readmitted within 30 days, **0** otherwise. The positive class makes up **11.2%** of the data, so this is a meaningfully imbalanced classification problem.
- **Cleaning:** `?` placeholders were converted to nulls. `weight` and `payer_code` were dropped for having very high missingness (97% and 40% respectively); remaining categorical nulls were filled with an explicit "Unknown"/"None" category rather than imputed, to avoid introducing information not actually available at discharge. Identifier columns (`encounter_id`, `patient_nbr`) were dropped before modeling.
- **Split:** 80/20 train/test, stratified on the target (81,412 / 20,354 encounters).

## Methodology

Three models were trained and compared, each inside a `ColumnTransformer` pipeline (one-hot encoding for categoricals, standard scaling for numerics):

| Step | What was tested |
|---|---|
| Baseline | Logistic Regression, unbalanced |
| Improvement | Logistic Regression, balanced class weights |
| Validation | 5-fold stratified cross-validation on the balanced model |
| Model 2 | Random Forest, balanced class weights, then hyperparameter tuning via `GridSearchCV` |
| Model 3 | HistGradientBoostingClassifier, balanced class weights |
| Final step | Threshold tuning on the best model, evaluated across a full precision/recall sweep |

## Results

| Model | ROC-AUC | Recall (class 1) | Precision (class 1) | F1 (class 1) |
|---|---|---|---|---|
| Logistic Regression (unbalanced, baseline) | 0.647 | 2% | 49% | 0.04 |
| Logistic Regression (balanced) | 0.644 | 55% | 17% | 0.26 |
| Random Forest (balanced) | 0.653 | 54% | 17% | 0.25 |
| **HistGradient Boosting (balanced)** | **0.675** | 61% | 18% | 0.27 |

Five-fold cross-validation on the balanced logistic regression model gave a mean ROC-AUC of **0.630** (range 0.626–0.638, std 0.004) — consistent with the single-split results and indicating stable, if modest, discriminatory power from this feature set.

**HistGradient Boosting was selected as the final model** based on ROC-AUC, the highest and most stable of the three.

### Threshold Tuning

Gradient boosting's default 0.5 threshold under-predicts the positive class relative to what the cost trade-off calls for. Sweeping thresholds from 0.05 to 0.85 and selecting the point that maximizes F1 for class 1 gives:

| Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| 0.50 (default) | 18% | 61% | 0.27 | 64% |
| **0.55 (selected)** | **21%** | **43%** | **0.28** | **76%** |

At the selected threshold, the model catches roughly **2 in 5 readmissions**, at the cost of flagging some patients who won't be readmitted — a trade-off deliberately favoring recall over precision, consistent with the higher cost of a missed readmission.

## Limitations & Next Steps

- **Threshold selection currently uses the test set.** The 0.55 threshold was chosen by evaluating candidate thresholds directly on `X_test`, which makes the reported metrics at that threshold a slightly optimistic estimate. The correct fix — splitting off a held-out validation set (or reusing cross-validation folds) to select the threshold, then evaluating once on an untouched test set — has been identified but not yet implemented.
- **`discharge_disposition_id` may include codes for death or hospice discharge.** Patients who fall into those categories cannot structurally be readmitted, and their inclusion may be adding noise to the non-readmitted class. Filtering these out is a natural next step.
- **Model discrimination tops out around 0.65–0.68 ROC-AUC**, suggesting the available features (largely demographic and utilization counts) capture only part of what drives readmission risk. Richer feature engineering — grouping ICD diagnosis codes into clinical categories, or engineering features from prior visit history — is a likely next lever.
- **This model is purely predictive, not causal.** It estimates *risk*, not the *effect* of any specific intervention. A natural extension would be estimating the causal effect of a targeted intervention (e.g., a discharge follow-up call) on readmission risk, using methods like propensity score matching or uplift modeling — turning a risk score into an actionable, effect-estimated recommendation.

## Tools

Python, pandas, scikit-learn (`ColumnTransformer`, `LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, `GridSearchCV`), matplotlib, seaborn.
