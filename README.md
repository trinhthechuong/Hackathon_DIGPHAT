# Hackathon: Multi-modal Immunotherapy Response Prediction in ccRCC

Welcome to the **Pharmacogenomic Day Hackathon**. In this challenge, you will develop a machine learning model to predict how patients with clear cell Renal Cell Carcinoma (ccRCC) respond to Immune Checkpoint Blockade (ICB) therapy.

This repository provides two primary notebooks to guide you through the process:
1. **01_Model_Development.ipynb**: Data exploration, preprocessing theory, and model optimization on the training set.
2. **02_Model_Deployment.ipynb**: Loading saved artifacts and the trained pipeline to generate predictions on an external, unseen test set.

---

# 1. Objective and Intended Use

## 1.1 Clinical Question
- **Primary Goal**: Can we predict the RECIST response (**Responder**: CR/PR vs. **Non-Responder**: SD/PD) using multi-modal pre-treatment data?
- **Target Population**: ccRCC patients receiving immunotherapy (e.g., Nivolumab vs. Everolimus).
- **Data Modalities**: Transcriptomic, Genomic, Proteomic (Deconvolution), and Clinical.

---

# 2. Dataset Overview

The dataset includes a rich variety of biological and clinical information:

- **Transcriptomic (RNA-seq)**: Gene-level TPM (Transcripts Per Million) counts.
- **Genomic (Somatic Mutations)**: Binary status or counts of mutations in frequent genes.
- **Immune Deconvolution**: Proportions of 22 immune cell types estimated via CIBERSORTx from bulk RNA-seq.
- **ssGSEA Pathway Scores**: Enrichment scores for Hallmark pathways (via GSEAPy).
- **Clinical Data**: Arm (Nivolumab/Everolimus), Age, Sex, MSKCC Risk Group, and previous treatment history.

### 2.1 Missing Data Visualiztion
We use the `missingno` library to visualize missingness patterns, especially in clinical variables where specific values like "NO_IF" are treated as `NaN`.

---

# 3. Outcome Definition

## 3.1 RECIST Mapping
We define a binary outcome for classification:
- **1 (Responder)**: Complete Response (CR) or Partial Response (PR).
- **0 (Non-Responder)**: Stable Disease (SD) or Progressive Disease (PD).

## 3.2 Class Imbalance Assessment
The cohort exhibits a natural imbalance (fewer responders). We handle this using **ADASYN** (Adaptive Synthetic Sampling) within our modeling pipeline.

---

# 4. Preprocessing Strategy (Leakage-Controlled)

Preprocessing is critical to avoid data leakage. All parameters (e.g., mean/variance for scaling, imputer kernels) are learned on the training set and applied to the test set.

## 4.1 Transcriptomic Data
- **Filtering**: Removal of low-expression genes based on variance.
- **Standardization**: Scaling gene features to zero mean and unit variance.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) retaining **95% of explained variance**.

## 4.2 Genomic (Mutation) Data
- **Feature Intersection**: Ensuring the same gene set is used across training and test datasets.

## 4.3 Immune Deconvolution
- **Centered-Logratio (CLR) Transformation**: Since cell proportions are compositional data, we apply CLR before standardization to handle the unit-sum constraint.

## 4.4 Pathway Scores & Clinical Data
- **Pathway Scaling**: Standardization of ssGSEA scores.
- **Clinical Encoding**:
  - **One-Hot**: Treatment Arm.
  - **Binary/Label**: Sex, Sarc, Rhab, Tumor Site.
  - **Ordinal**: MSKCC Risk, Number of Prior Therapies.
  - **Standardization**: Age.
- **Imputation**: Multivariate Imputation by Chained Equations (**MICE**) via `miceforest`.

---

# 5. Feature Engineering and Selection

Given the high-dimensional nature of omics data:
1. **Variance Filtering**: Removes near-constant features.
2. **SelectKBest**: Uses **Mutual Information** to select the most predictive features within a cross-validation loop.
3. **Pipeline Integration**: Imbalance handling (ADASYN), Feature Selection (SelectKBest), and Modeling (Logistic Regression) are all wrapped in a single `Pipeline`.

---

# 6. Model Development and Evaluation

## 6.1 Baseline Model
We utilize **Penalized Logistic Regression (L2)** for its interpretability and robust performance on biological data.

## 6.2 Evaluation Metrics
Models are evaluated using 5-fold **Repeated Stratified Cross-Validation** with:
- **ROC AUC**: Overall discriminative ability.
- **PR AUC**: Performance on the minority (Responder) class.
- **Balanced Accuracy**: Accounting for class imbalance.
- **MCC / F1-Score**: Robustness of binary predictions.

## 6.3 Learning Curves
We plot training vs. validation performance to diagnose over-fitting and assess if more data would improve the model.

---

# 7. Model Deployment and Submission

## 7.1 Artifact Locking
All preprocessing objects (scalers, encoders, PCA, imputer) and the final pipeline are saved as picklable objects. The **Deployment Notebook** reloads these to ensure the test data is handled identically to the training data.

## 7.2 automated Submission
Instead of manual uploads, students submit results directly via an HTTP POST request to a Google Form "Invisible API":
1. Generate probabilities and binary predictions.
2. Format predictions as a comma-separated string.
3. Run the submission cell with your **Team Name**.

---

## 🎉 Getting Started
1. Open `01_Model_Development.ipynb` to explore the data.
2. Build your pipeline and optimize your parameters.
3. Use `02_Model_Deployment.ipynb` for final validation and submission.

**Good luck and happy hacking!**
