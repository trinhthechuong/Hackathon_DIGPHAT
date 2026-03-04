# Hackathon_DIGPHAT
PharmacogenomicDay
# NOTEBOOK TEMPLATE

You are a good Bioinformatic assistant. Me and you, we will teach students in a hackathon how to develop a machine learning model using pharmacogenomic data.

Your tasks are help me generating two notebooks:

1. The first one called **Model development**, where I present the datasets and theory to the students alongside with the code.
2. The second one called **Model deployment**. After the student satisfy with the model performance, they will use this notebook to load the model and all the material that they developed on training dataset (retaining features, normalization scaler, etc) will be applied on the external test set to make a prediction.

---

# Use Case

**Multi-modal prediction of Immunotherapy response for ccRCC patient with follow-up post-Immune Checkpoint Blockade.**

The datasets are provided in the folder.

---

# 1. Objective and Intended Use

In this step, you should generate the information or explanation of each section.  
For example, you can show the first 5 lines of RNA-seq dataset and give a short description about it.

## 1.1 Clinical Question

- Can we predict RECIST response (CR/PR vs SD/PD) to immunotherapy?
- Target population:
  - Specify tumor type
  - Treatment class
  - Metastases + primary sites

---

# 2. Dataset Overview

## 2.1 Modalities Included

- RNA-seq (gene-level TPM)
- Somatic mutation status (frequent genes)
- Clinical/biological variables
- Immune cell proportions (deconvolution):  
  Using Cibersortx to estimation cell type proportion from bulk RNA-seq.  
  I will provide directly a data, don’t need to generate a code to have this data.
- ssGSEA pathway scores: ssGSEA from GSEAPy Hallmark pathways

---

## 2.2 Cohort Description

- Number of patients
- Responder proportion
- Missing data summary:
  - Using `missingno` library to show heatmap

---

## 2.3 Data Harmonization

- Sample ID intersection
- Dataset freeze version (define once) → reproducibility (tool version...)

---

# 3. Outcome Definition

## 3.1 RECIST Mapping

Binary outcome:

- 1 = CR/PR
- 0 = SD/PD

Justification stated clearly.

---

## 3.2 Class Imbalance Assessment

- Ratio responders / non-responders
- Visualize by bar chart

---

# 4. Preprocessing Strategy (Leakage-Controlled)

Explicit statement: you should access `/Users/thechuongtrinh/Workspace/Hackathon_DIGPHAT/Data/preprocessing.py` and `/Users/thechuongtrinh/Workspace/Hackathon_DIGPHAT/Data/gene_query.py` to understand the preprocessing steps and follow them.

All preprocessing steps are done on the data.

## 4.1 RNA-seq
- Gene name harmonization in from Ensembl ID to Gene Symbol
- Normalization (CPM / VST / log-transform):
  - Mention the libraries but don’t need to generate the code because the data is already normalized to TPM
- Filtering low-expression genes
- Optional variance filtering
- Scaling

---

## 4.2 Mutation Data

- Binary encoding

---

## 4.3 Immune Deconvolution

- Log-scaling because they are compositional data

---

## 4.4 Pathway Scores

- Scaling
- Redundancy awareness

---

## 4.5 Clinical Data

- Encoding categorical variables
- Standardization

---

## 4.6 Missing Value Handling

- Introduce basic technique for imputation
- Apply multivariate imputation using MICE (`miceforest`)

---

# 6. Feature Engineering and Dimensionality Reduction

Given high-dimensional transcriptomics:

## 6.1 Filtering-Based Reduction / Feature Selection

- Variance filtering
- Optional univariate filtering (responder information)

---

## 6.2 Embedded Selection

- Regularization L1
- Elastic Net

---

## 6.3 Dimensionality Reduction

- PCA (transcriptome only)
- Pre-specified number of components

---

# 7. Model Development

## 7.1 Baseline Model

- Penalized logistic regression
- Provides interpretability and calibration

---

## 7.2 Non-Linear Model

- Decision-tree
- Random Forest
- SVM
- XG-Boost
- MLP

---

## 7.3 Cross-Validation

- Stratified k-fold on training set

---

# 8. Model Evaluation

## 8.1 Metrics

- PR AUC
- Balanced accuracy
- F1 score
- MCC
- Confusion matrix

## 8.3 Learning Curve

- Over-fitting / under-sampling

---

# 9. Idea for Model Improvement

- Data sampling (SMOTE)
- Choice of ML model
- Hyperparameter tuning (GridSearchCV, Optuna)

---

# 9. Biological Interpretation (If Time Allows)

## 9.1 Global Feature Importance

- Top predictive variables
- Stability across folds (qualitative)

---

# Deployment Notebook

---

# 11. External Validation Preparation (Blind Phase Ready)

In this step, we will load all scalers, selected features, or other artifacts and apply them on external dataset to make a prediction.

---

## 11.1 Model Locking

Freeze:

- Selected features
- Scalers
- Model coefficients

Save model artifact.

---

## 11.2 External Dataset Pipeline

Predefined steps:

- Align feature names
- Apply stored preprocessing
- Generate probabilities
- Generate binary predictions

---

## 11.3 Performance Reporting Template

When labels become available:

- ROC AUC
- PR AUC
- MCC
- Calibration
- Confusion matrix

---

# Using Google Form as Invisible API Database

Students won't ever see the form UI; they will simply push their results directly from their Python notebook code.

---

## Step 1: Create the Database (Google Form)

1. Create a new Google Form.
2. Add two Short Answer questions:
   - Team Name
   - Predictions (where they paste the string of labels)
3. Go to Responses tab and click the green Google Sheets icon to create spreadsheet.

---

## Step 2: Get the Hidden Entry IDs

1. In Form editor, click three dots (top right) → Get pre-filled link.
2. Type "TEST_TEAM" in first box and "TEST_LABELS" in second.
3. Click Get link and copy it.

Example:
