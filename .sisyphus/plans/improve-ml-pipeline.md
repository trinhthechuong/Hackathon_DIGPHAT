# Improve ML Pipeline for ccRCC Immunotherapy Response Prediction

## TL;DR

> **Quick Summary**: The current model (ADASYN + SelectKBest + LogisticRegression L2) achieves ROC-AUC 0.519 (essentially random) on the held-out test set. We will append ~15 new cells to the end of `01_Model_Development.ipynb` that build a proper ML pipeline with genomic data integration, aggressive feature selection (k<=25 features for 137 samples), multiple classifiers (L1 LogReg, linear SVC, XGBoost), cross-validation, and SHAP visualizations -- targeting ROC-AUC 0.65-0.75.
> 
> **Deliverables**:
> - New "Improved Model Development" section appended to `01_Model_Development.ipynb`
> - Genomic data integrated (34 mutation/CNV features encoded)
> - 3 classifiers compared with honest CV metrics
> - SHAP beeswarm + ROC curve + confusion matrix + comparison bar chart
> - Improved pipeline artifacts saved to `artifacts/pipeline_artifacts_improved.joblib`
> 
> **Estimated Effort**: Medium (3-4 hours implementation)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5 -> Task 6 -> Task 7 -> Task 8 -> Task 9

---

## Context

### Original Request
"I am preparing a hackathon for my student and I will demonstrate the machine learning model predicting responders and non-responders. Right now, the results are poor and I need a reasonable ROC-AUC and f1-score. Could you please help me to find the best pipeline or model."

### Interview Summary
**Key Discussions**:
- **Implementation approach**: Add new cells at the end of the existing notebook (preserve educational content for students to see before/after comparison)
- **Scope**: Focus ONLY on `01_Model_Development.ipynb` -- NOT updating `02_Model_Deployment.ipynb`
- **Visualization**: Include SHAP + plots for impressive hackathon demo
- **No formal test framework**: Testing is interactive in notebooks

**Research Findings**:
- L1 Logistic Regression + SVM proven best for small N high-dimensional biomedical data
- Feature-to-sample ratio should be N/10 to N/5 -> 14-27 features for N=137
- BorderlineSMOTE best for N<100 BUT `class_weight='balanced'` is simpler and avoids k-neighbor failures in small CV folds
- XGBoost `scale_pos_weight` outperforms SMOTE for small samples (research-confirmed)
- Nested CV mandatory for unbiased estimates
- Known ccRCC biomarkers (BAP1, PBRM1, VHL, SETD2) present in unused genomic data
- RFECV does NOT work inside imblearn Pipeline -- use SelectKBest instead
- SHAP does not work natively with VotingClassifier -- use individual best model

### Metis Review
**Identified Gaps** (addressed):
- **Artifact collision**: Existing `pipeline_artifacts.joblib` used by deployment notebook -> save improved artifacts to `pipeline_artifacts_improved.joblib` (NEVER overwrite)
- **Genomic NO_IF**: 37% of patients have ALL NO_IF -> use `has_genomic_data` binary indicator + simple imputation, check distribution vs target
- **PCA vs interpretability**: PCA gives uninterpretable "PC42" for SHAP -> use supervised gene selection from raw transcriptomics instead
- **VotingClassifier incompatible with SHAP**: Drop VotingClassifier, use best single model for SHAP
- **SMOTE k-neighbor failures**: Use `class_weight='balanced'` / `scale_pos_weight` instead of SMOTE/ADASYN
- **Mutual info non-deterministic**: Use `functools.partial(mutual_info_classif, random_state=42)`
- **XGBoost double-compensation**: NEVER combine `scale_pos_weight` with SMOTE
- **joblib save failure**: Set `n_jobs=None` before `joblib.dump()`

---

## Work Objectives

### Core Objective
Append new cells to `01_Model_Development.ipynb` that improve the ML pipeline from ROC-AUC ~0.52 to >=0.60 (target 0.65-0.75) with proper feature selection, multiple classifiers, honest CV, and impressive SHAP visualizations for hackathon demo.

### Concrete Deliverables
- New markdown + code cells appended after existing notebook content
- Genomic data loaded, encoded (MUT->1, WT->0, NO_IF->NaN), and integrated
- Combined feature matrix with <=25 features via SelectKBest
- 3 classifier pipelines evaluated with RepeatedStratifiedKFold CV
- Best model selected and SHAP plots generated
- Improved artifacts saved to `artifacts/pipeline_artifacts_improved.joblib`

### Definition of Done
- [ ] `jupyter nbconvert --execute 01_Model_Development.ipynb` runs without error
- [ ] New cells print ROC-AUC >= 0.60 (mean CV)
- [ ] SHAP beeswarm plot shows real feature names (not PC numbers)
- [ ] 4 visualization plots render successfully
- [ ] Existing notebook cells are UNMODIFIED (byte-identical)
- [ ] `artifacts/pipeline_artifacts_improved.joblib` is loadable

### Must Have
- Genomic data integrated with proper encoding
- <=25 features in final model (N/5 rule)
- At least 3 classifiers compared (L1 LogReg, linear SVC, XGBoost)
- RepeatedStratifiedKFold cross-validation (5-fold, 3+ repeats)
- SHAP feature importance visualization with biological feature names
- ROC curve plot
- Confusion matrix plot
- Model comparison bar chart
- Improved artifacts saved (separate file from existing)
- `random_state=42` everywhere for reproducibility
- `class_weight='balanced'` or `scale_pos_weight` for imbalance (NOT SMOTE)

### Must NOT Have (Guardrails)
- G1: **Zero modification** to existing notebook cells -- students must see before/after
- G2: Save improved artifacts to `pipeline_artifacts_improved.joblib` -- NEVER overwrite existing `pipeline_artifacts.joblib`
- G3: **<=25 features** maximum in all models
- G4: **<=3 classifiers** (L1 LogReg, linear SVC, XGBoost) -- no neural nets, RF, LightGBM
- G5: **<=4 visualization plots** total (comparison bar, ROC, SHAP beeswarm, confusion matrix)
- G6: **No VotingClassifier or StackingClassifier** -- incompatible with SHAP, adds complexity
- G7: **No SMOTE/ADASYN/BorderlineSMOTE** -- use class_weight/scale_pos_weight instead (avoids k-neighbor failures in small folds)
- G8: **No RFECV** -- does not work inside imblearn Pipeline per research
- G9: **No KernelExplainer** -- use TreeExplainer (XGBoost) or LinearExplainer (LogReg/SVC) only
- G10: **No new feature engineering** (no interaction terms, polynomials, autoencoders)
- G11: **No modification** to `02_Model_Deployment.ipynb`
- G12: **No `n_jobs=-1`** in saved objects -- set `n_jobs=None` before joblib.dump
- G13: Use `functools.partial(mutual_info_classif, random_state=42)` for deterministic feature selection

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** -- ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (Jupyter notebooks only, no pytest)
- **Automated tests**: None -- verification via notebook execution and output inspection
- **Framework**: None

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Notebook cells**: Use Bash (`jupyter nbconvert --execute` or `python -c "import nbformat; ..."`) to validate
- **Model metrics**: Parse printed output from notebook cells
- **Visualizations**: Verify cell execution does not error; check for saved figure files if applicable
- **Artifacts**: Use `python -c "import joblib; ..."` to validate loadability

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Sequential Foundation -- must be in order):
+-- Task 1: Verify dependencies and install if missing [quick]
+-- Task 2: Read notebook + locate insertion point + understand existing variables [quick]
+-- Task 3: Encode genomic data + build combined feature matrix cells [deep]

Wave 2 (Core Implementation -- sequential, builds on Wave 1):
+-- Task 4: Build 3 classifier pipelines + CV comparison cells [deep]
+-- Task 5: Best model selection + SHAP visualization cells [deep]
+-- Task 6: ROC curve + confusion matrix + comparison bar chart cells [visual-engineering]

Wave 3 (Finalization -- sequential):
+-- Task 7: Save improved artifacts cell [quick]
+-- Task 8: Validate notebook JSON integrity + execute all new cells [deep]
+-- Task 9: Final acceptance criteria verification [deep]

Wave FINAL (After ALL tasks -- independent review, 4 parallel):
+-- Task F1: Plan compliance audit (oracle)
+-- Task F2: Code quality review (unspecified-high)
+-- Task F3: Real manual QA (unspecified-high)
+-- Task F4: Scope fidelity check (deep)

Critical Path: Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5 -> Task 6 -> Task 7 -> Task 8 -> Task 9 -> F1-F4
Parallel Speedup: Limited (notebook is inherently sequential -- each cell depends on prior)
Max Concurrent: 4 (Final verification wave only)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | -- | 2, 3, 4, 5, 6 | 1 |
| 2 | 1 | 3 | 1 |
| 3 | 2 | 4 | 1 |
| 4 | 3 | 5 | 2 |
| 5 | 4 | 6 | 2 |
| 6 | 5 | 7 | 2 |
| 7 | 6 | 8 | 3 |
| 8 | 7 | 9 | 3 |
| 9 | 8 | F1-F4 | 3 |
| F1-F4 | 9 | -- | FINAL |

### Agent Dispatch Summary

- **Wave 1**: 3 tasks -- T1 -> `quick`, T2 -> `quick`, T3 -> `deep`
- **Wave 2**: 3 tasks -- T4 -> `deep`, T5 -> `deep`, T6 -> `visual-engineering`
- **Wave 3**: 3 tasks -- T7 -> `quick`, T8 -> `deep`, T9 -> `deep`
- **Wave FINAL**: 4 tasks -- F1 -> `oracle`, F2 -> `unspecified-high`, F3 -> `unspecified-high`, F4 -> `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.
> **A task WITHOUT QA Scenarios is INCOMPLETE. No exceptions.**

- [x] 1. Verify Dependencies and Install Missing Packages

  **What to do**:
  - Run `python -c "import shap; import xgboost; print('OK')"` to check if SHAP and XGBoost are installed
  - If either is missing, install via `pip install shap xgboost` (use `!pip install` in notebook context)
  - Verify SHAP version >= 0.40 (for TreeExplainer compatibility)
  - Verify xgboost version >= 1.5 (for `scale_pos_weight` support)
  - Append a markdown cell to notebook: `## 6. Improved Model Development` with introductory text explaining we will now try a better pipeline
  - Append a code cell that runs the dependency check and prints versions

  **Must NOT do**:
  - Do NOT modify any existing cells
  - Do NOT install unnecessary packages

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple dependency check and install, minimal logic
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (sequential start)
  - **Blocks**: Tasks 2, 3, 4, 5, 6, 7
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `01_Model_Development.ipynb` cell execution_count=59 -- existing import cell pattern (imports + prints 'All imports OK'). Follow same pattern for new dependency check cell.

  **External References**:
  - SHAP docs: https://shap.readthedocs.io/ -- verify TreeExplainer and LinearExplainer APIs
  - XGBoost docs: https://xgboost.readthedocs.io/ -- verify XGBClassifier API

  **WHY Each Reference Matters**:
  - The existing import cell pattern shows how the notebook handles dependency verification (try/except + print confirmation)

  **Acceptance Criteria**:
  - [ ] `python -c "import shap; import xgboost; print('OK')"` prints OK without error
  - [ ] New markdown cell added at end of notebook with `## 6. Improved Model Development`
  - [ ] New code cell added that prints SHAP and XGBoost versions

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Dependencies available
    Tool: Bash
    Preconditions: Python environment active
    Steps:
      1. Run: python -c "import shap; print(shap.__version__); import xgboost; print(xgboost.__version__)"
      2. Assert both version strings are printed without ImportError
    Expected Result: Two version strings printed (e.g. '0.44.1' and '2.0.3')
    Failure Indicators: ImportError or ModuleNotFoundError
    Evidence: .sisyphus/evidence/task-1-deps-check.txt

  Scenario: Notebook has new section header
    Tool: Bash
    Preconditions: Task 1 completed
    Steps:
      1. Run: python -c "import nbformat; nb=nbformat.read('01_Model_Development.ipynb',4); last_md=[c for c in nb.cells if c.cell_type=='markdown'][-1]; print(last_md.source[:50])"
      2. Assert output contains '## 6. Improved Model Development'
    Expected Result: Last markdown cell starts with '## 6. Improved Model Development'
    Failure Indicators: Different text or cell not found
    Evidence: .sisyphus/evidence/task-1-section-header.txt
  ```

  **Commit**: NO (groups with Task 9)

- [ ] 2. Read Notebook Structure and Map Existing Variables

  **What to do**:
  - Read the ENTIRE `01_Model_Development.ipynb` notebook to understand:
    - Total number of cells (to know exact insertion index for new cells)
    - All variables available in kernel memory at end of notebook execution:
      - `X_train_all` (137 x ~37930) -- full feature matrix
      - `y_train` (137,) -- binary labels (31 R, 106 NR)
      - `train_clinical_imputed` -- clinical features (137 x 8, includes Response)
      - `train_trans_scaled` -- scaled transcriptomic (137 x 37854)
      - `train_pathway_scaled` -- scaled ssGSEA pathways (137 x 50)
      - `deconv_clr_df` -- CLR-transformed deconvolution (137 x 22)
      - `pca`, `pca_columns` -- fitted PCA object and column names
      - `keep_genes` -- list of variance-filtered gene names
      - `trans_scaler`, `pathway_scaler`, `clr_scaler` -- fitted scalers
      - `transformers` dict -- stores all fitted preprocessing objects
      - `RANDOM_STATE = 42`
      - `cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)`
    - Identify the LAST cell index for appending new cells
    - Confirm genomic data file path: `Data/train_nivo/genomic.csv`
    - Note: Genomic data was displayed in section 2 but NEVER loaded into a variable for modeling
  - Document findings so subsequent tasks know exact variable names and shapes to reference

  **Must NOT do**:
  - Do NOT modify any cells
  - Do NOT execute the notebook

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Read-only analysis of notebook structure
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (after Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:
  - `01_Model_Development.ipynb` -- ENTIRE file. Read all cells to map variables.
  - `Data/train_nivo/genomic.csv` -- genomic data file (137 rows, 34 columns)

  **WHY Each Reference Matters**:
  - The notebook must be read completely to identify what variables exist at the end of execution
  - Genomic CSV must be checked for exact column names and value encodings

  **Acceptance Criteria**:
  - [ ] Complete variable map documented (names, shapes, types)
  - [ ] Last cell index identified for insertion point
  - [ ] Genomic CSV column names and value encodings confirmed

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Variable map is complete
    Tool: Bash
    Preconditions: Notebook file exists
    Steps:
      1. Run: python -c "import nbformat; nb=nbformat.read('01_Model_Development.ipynb',4); print(f'Total cells: {len(nb.cells)}')"
      2. Assert cell count is printed (expected ~90-120 cells)
    Expected Result: Number of cells printed
    Failure Indicators: File read error
    Evidence: .sisyphus/evidence/task-2-notebook-structure.txt
  ```

  **Commit**: NO (groups with Task 9)

- [ ] 3. Encode Genomic Data and Build Combined Feature Matrix

  **What to do**:
  Append the following NEW CELLS to the notebook (after the section header from Task 1):

  **Cell 3a (markdown)**: `### 6.1 Genomic Data Integration`
  Explain: encoding MUT=1, WT=0, NO_IF=NaN, then filling NaN with 0 (WT default), adding `has_genomic_data` indicator.

  **Cell 3b (code)**: Load and encode genomic data:
  - Load `Data/train_nivo/genomic.csv`
  - Drop Patient_ID
  - Replace MUT->1, WT->0, NO_IF->NaN, cast to float
  - Add `has_genomic_data` indicator (0 if ALL features are NaN for that patient)
  - Fill remaining NaN with 0 (WT is the most common/default)
  - Print shape and number of patients with genomic data

  **Cell 3c (markdown)**: `### 6.2 Combined Feature Matrix (Without PCA)`
  Explain: combining clinical + pathway + deconvolution + genomic = ~114 features. NOT using PCA transcriptomics to keep features interpretable for SHAP.

  **Cell 3d (code)**: Build combined feature matrix:
  - Get clinical features from `train_clinical_imputed` -- MUST drop 'Response' column
  - Concatenate: clinical + `train_pathway_scaled` + `deconv_clr_df` + genomic_filled
  - Reset indices before concat to avoid alignment issues
  - Store `feature_names = list(X_combined.columns)` for SHAP later
  - Print feature breakdown by modality

  **CRITICAL NOTES**:
  - `train_clinical_imputed` includes 'Response' column -- MUST drop it (data leakage!)
  - DO NOT use PCA transcriptomics -- we want interpretable feature names for SHAP
  - Genomic columns include known ccRCC biomarkers: BAP1, PBRM1, VHL, SETD2

  **Must NOT do**:
  - Do NOT use PCA transcriptomics (G10 -- keeps features interpretable)
  - Do NOT modify existing cells (G1)
  - Do NOT create interaction terms or polynomial features (G10)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex data integration with multiple modalities, careful variable handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (after Task 2)
  - **Blocks**: Task 4
  - **Blocked By**: Task 2

  **References**:

  **Pattern References**:
  - `01_Model_Development.ipynb` cells around execution_count=14-16 -- how existing modalities are loaded and concatenated (shows reset_index pattern)
  - `01_Model_Development.ipynb` section 3 -- how genomic data is displayed but NOT encoded

  **API/Type References**:
  - `train_clinical_imputed` DataFrame -- has 8 columns including 'Response'. MUST drop Response.
  - `train_pathway_scaled` DataFrame -- 50 pathway columns, already scaled
  - `deconv_clr_df` DataFrame -- 22 immune cell columns, already CLR-transformed
  - `y_train` -- 137-element Series with binary labels

  **External References**:
  - Braun et al. (2020) Nature Medicine -- BAP1, PBRM1, VHL, SETD2 are known ccRCC biomarkers in genomic data

  **WHY Each Reference Matters**:
  - Existing concat pattern (cell ~14-16) shows reset_index usage -- critical for alignment
  - `train_clinical_imputed` MUST have Response dropped -- most likely leakage bug

  **Acceptance Criteria**:
  - [ ] Genomic data encoded: MUT=1, WT=0, NO_IF=NaN, then filled
  - [ ] `has_genomic_data` indicator column present
  - [ ] Combined matrix shape is (137, ~114)
  - [ ] NO 'Response' column in feature matrix
  - [ ] Feature names stored for SHAP

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Combined matrix has correct shape and no leakage
    Tool: Bash
    Preconditions: Cells 3b and 3d have been appended to notebook
    Steps:
      1. Execute the new cells or full notebook
      2. Check printed output for 'Combined feature matrix: (137, N)' where N is ~114
      3. Search new cell source for 'Response' -- must only appear in drop() call
    Expected Result: Shape (137, ~114), no Response leakage
    Failure Indicators: Response column in features, wrong shape
    Evidence: .sisyphus/evidence/task-3-combined-matrix.txt

  Scenario: Genomic encoding is correct
    Tool: Bash
    Preconditions: Cell 3b appended and executable
    Steps:
      1. Check printed output for 'Patients with genomic data: ~86/137'
      2. Verify has_genomic_data column exists
    Expected Result: ~86 patients have genomic data, indicator column present
    Failure Indicators: 0 or 137 patients (encoding bug)
    Evidence: .sisyphus/evidence/task-3-genomic-encoding.txt
  ```

  **Commit**: NO (groups with Task 9)

- [ ] 4. Build 3 Classifier Pipelines and CV Comparison

  **What to do**:
  Append the following NEW CELLS to the notebook (after cells from Task 3):

  **Cell 4a (markdown)**: `### 6.3 Model Comparison: 3 Classifiers with Nested Feature Selection`
  Explain: comparing L1 LogReg, linear SVC, and XGBoost with SelectKBest(k=20, mutual_info) inside each pipeline. Using class_weight/scale_pos_weight instead of SMOTE. RepeatedStratifiedKFold for honest evaluation.

  **Cell 4b (code)**: Build 3 sklearn Pipelines and evaluate:
  ```python
  import functools
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import StandardScaler
  from sklearn.feature_selection import SelectKBest, mutual_info_classif
  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC
  from xgboost import XGBClassifier
  from sklearn.model_selection import cross_validate
  import numpy as np
  import pandas as pd
  
  mi_scorer = functools.partial(mutual_info_classif, random_state=42)
  
  pipelines = {
      'L1 LogReg': Pipeline([
          ('scaler', StandardScaler()),
          ('selector', SelectKBest(score_func=mi_scorer, k=20)),
          ('clf', LogisticRegression(
              penalty='l1', solver='liblinear', C=0.1,
              class_weight='balanced', random_state=42, max_iter=1000
          ))
      ]),
      'Linear SVC': Pipeline([
          ('scaler', StandardScaler()),
          ('selector', SelectKBest(score_func=mi_scorer, k=20)),
          ('clf', SVC(
              kernel='linear', C=0.1, class_weight='balanced',
              probability=True, random_state=42
          ))
      ]),
      'XGBoost': Pipeline([
          ('scaler', StandardScaler()),
          ('selector', SelectKBest(score_func=mi_scorer, k=20)),
          ('clf', XGBClassifier(
              n_estimators=50, max_depth=3, learning_rate=0.01,
              scale_pos_weight=3.42, random_state=42,
              eval_metric='logloss', use_label_encoder=False
          ))
      ])
  }
  
  scoring = ['roc_auc', 'f1', 'balanced_accuracy']
  results = {}
  for name, pipe in pipelines.items():
      cv_results = cross_validate(pipe, X_combined, y_train, cv=cv,
                                  scoring=scoring, return_train_score=False)
      results[name] = {
          'ROC-AUC': f"{cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}",
          'F1': f"{cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}",
          'Bal. Acc.': f"{cv_results['test_balanced_accuracy'].mean():.3f} ± {cv_results['test_balanced_accuracy'].std():.3f}",
          'roc_auc_mean': cv_results['test_roc_auc'].mean(),  # for selecting best
      }
      print(f"{name}: ROC-AUC={results[name]['ROC-AUC']}, F1={results[name]['F1']}")
  
  results_df = pd.DataFrame(results).T[['ROC-AUC', 'F1', 'Bal. Acc.']]
  print('\n', results_df)
  ```
  - Use `X_combined` (from Task 3) and `y_train` as data
  - Use `cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)` (already in memory)
  - `mi_scorer` uses `functools.partial(mutual_info_classif, random_state=42)` for determinism (G13)
  - k=20 features (within <=25 guardrail G3)
  - NO SMOTE/ADASYN (G7) -- uses class_weight='balanced' and scale_pos_weight=3.42
  - scale_pos_weight = 106/31 ≈ 3.42 (class ratio)
  - Store raw `roc_auc_mean` for best model selection in Task 5

  **Must NOT do**:
  - Do NOT use SMOTE/ADASYN/BorderlineSMOTE (G7)
  - Do NOT use RFECV (G8)
  - Do NOT use more than 3 classifiers (G4)
  - Do NOT use VotingClassifier or StackingClassifier (G6)
  - Do NOT modify existing cells (G1)
  - Do NOT use k > 25 in SelectKBest (G3)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex pipeline construction with multiple classifiers, cross-validation, and careful hyperparameter choices. Requires understanding of sklearn pipeline API and data flow.
  - **Skills**: []
    - No specialized skills needed (pure sklearn/xgboost code)
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser interaction needed
    - `frontend-ui-ux`: No UI work

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `01_Model_Development.ipynb` cells around execution_count=51-55 -- existing pipeline construction (ADASYN + SelectKBest + LogReg). Shows Pipeline() syntax and how SelectKBest is used. NEW pipelines replace ADASYN with class_weight.
  - `01_Model_Development.ipynb` cells around execution_count=56-58 -- existing cross_validate call pattern. Shows scoring parameter usage and result extraction.

  **API/Type References**:
  - `X_combined` DataFrame from Task 3 -- shape (137, ~114), all float64
  - `y_train` Series -- 137 binary labels (31 R, 106 NR)
  - `cv` -- RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42), already in kernel memory
  - `feature_names` list from Task 3 -- column names for SHAP in Task 5

  **External References**:
  - sklearn SelectKBest docs: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
  - XGBClassifier docs: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
  - functools.partial for mutual_info_classif: ensures random_state=42 is passed through SelectKBest

  **WHY Each Reference Matters**:
  - Existing pipeline cells show the exact Pipeline() construction pattern used in this notebook -- follow the same import/construction style for consistency
  - Existing cross_validate cells show how to extract and print scores -- follow same print format
  - X_combined and y_train are the EXACT variable names to use (from Task 3)
  - feature_names is needed to map SelectKBest indices back to real column names in Task 5

  **Acceptance Criteria**:
  - [ ] 3 pipelines constructed: L1 LogReg, Linear SVC, XGBoost
  - [ ] Each pipeline uses StandardScaler -> SelectKBest(k=20) -> Classifier
  - [ ] cross_validate runs with scoring=['roc_auc', 'f1', 'balanced_accuracy']
  - [ ] Results table printed with mean ± std for each metric
  - [ ] No SMOTE/ADASYN in any pipeline
  - [ ] k=20 in SelectKBest (<=25 guardrail)
  - [ ] `functools.partial(mutual_info_classif, random_state=42)` used for determinism

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: All 3 models produce valid CV metrics
    Tool: Bash
    Preconditions: Cells 4a-4b appended and X_combined, y_train available
    Steps:
      1. Execute notebook or run the cell code in Python
      2. Check printed output for 3 model names with ROC-AUC values
      3. Assert all ROC-AUC values are between 0.0 and 1.0 (not NaN)
      4. Assert at least one model has ROC-AUC >= 0.55 (above random)
    Expected Result: 3 rows of metrics printed, all valid numbers, at least one >= 0.55
    Failure Indicators: NaN values, ValueError, Pipeline construction error, all models at 0.5
    Evidence: .sisyphus/evidence/task-4-cv-metrics.txt

  Scenario: No forbidden oversampling in pipelines
    Tool: Bash
    Preconditions: Cell 4b appended
    Steps:
      1. Run: python -c "import nbformat; nb=nbformat.read('01_Model_Development.ipynb',4); [print(c.source) for c in nb.cells if 'ADASYN' in c.source or 'SMOTE' in c.source or 'BorderlineSMOTE' in c.source]"
      2. Assert no NEW cells contain ADASYN/SMOTE (existing cells may)
    Expected Result: Only existing cells reference ADASYN, no new cells
    Failure Indicators: New cell with ADASYN/SMOTE import or usage
    Evidence: .sisyphus/evidence/task-4-no-smote.txt
  ```

  **Commit**: NO (groups with Task 9)

- [ ] 5. Best Model Selection and SHAP Feature Importance Visualization

  **What to do**:
  Append the following NEW CELLS to the notebook (after cells from Task 4):

  **Cell 5a (markdown)**: `### 6.4 Best Model & SHAP Feature Importance`
  Explain: selecting the best model by highest mean ROC-AUC, fitting on full training data, then using SHAP to explain which features drive predictions.

  **Cell 5b (code)**: Select best model and generate SHAP:
  ```python
  import shap
  
  # Select best model by ROC-AUC
  best_name = max(results, key=lambda k: results[k]['roc_auc_mean'])
  best_pipeline = pipelines[best_name]
  print(f'Best model: {best_name} (ROC-AUC: {results[best_name]["ROC-AUC"]})')
  
  # Fit best pipeline on full training data
  best_pipeline.fit(X_combined, y_train)
  
  # Extract selected feature names
  selector = best_pipeline.named_steps['selector']
  selected_mask = selector.get_support()
  selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
  print(f'Selected {len(selected_features)} features: {selected_features}')
  
  # Get transformed data for SHAP
  X_scaled = best_pipeline.named_steps['scaler'].transform(X_combined)
  X_selected = best_pipeline.named_steps['selector'].transform(X_scaled)
  X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
  
  # SHAP explainer (choose based on model type)
  clf = best_pipeline.named_steps['clf']
  if hasattr(clf, 'feature_importances_'):  # XGBoost
      explainer = shap.TreeExplainer(clf)
  else:  # LogReg or SVC
      explainer = shap.LinearExplainer(clf, X_selected_df)
  
  shap_values = explainer.shap_values(X_selected_df)
  
  # SHAP beeswarm plot
  plt.figure(figsize=(10, 8))
  shap.summary_plot(shap_values, X_selected_df, show=False)
  plt.title(f'SHAP Feature Importance ({best_name})')
  plt.tight_layout()
  plt.savefig('artifacts/shap_beeswarm.png', dpi=150, bbox_inches='tight')
  plt.show()
  
  # Top 10 features
  mean_abs_shap = np.abs(shap_values).mean(axis=0)
  top_10_idx = np.argsort(mean_abs_shap)[::-1][:10]
  print('\nTop 10 most important features:')
  for i, idx in enumerate(top_10_idx):
      print(f'  {i+1}. {selected_features[idx]}: {mean_abs_shap[idx]:.4f}')
  ```
  - SHAP explainer choice: TreeExplainer for XGBoost, LinearExplainer for LogReg/SVC (G9 -- no KernelExplainer)
  - Uses `feature_names` from Task 3 to map indices back to real biological names
  - Saves beeswarm plot to `artifacts/shap_beeswarm.png`
  - Prints top 10 features for hackathon discussion

  **Must NOT do**:
  - Do NOT use KernelExplainer (G9 -- slow and unnecessary)
  - Do NOT use VotingClassifier for SHAP (G6)
  - Do NOT modify existing cells (G1)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: SHAP API requires careful handling -- different explainer types for different models, feature name mapping through pipeline, and proper plot configuration.
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser needed for SHAP generation
    - `frontend-ui-ux`: matplotlib plots, not web UI

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `01_Model_Development.ipynb` cells around execution_count=55 -- existing pipeline.fit() call. Shows how to fit pipeline on training data.
  - `01_Model_Development.ipynb` cells using `plt.show()` and `plt.savefig()` -- existing plot patterns for consistent style.

  **API/Type References**:
  - `results` dict from Task 4 -- contains 'roc_auc_mean' key for each model name
  - `pipelines` dict from Task 4 -- maps model name to fitted Pipeline object
  - `X_combined` DataFrame from Task 3 -- input to pipeline.fit()
  - `feature_names` list from Task 3 -- maps SelectKBest indices to column names
  - `y_train` Series -- labels for fitting

  **External References**:
  - SHAP docs (TreeExplainer): https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
  - SHAP docs (LinearExplainer): https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html
  - SHAP summary_plot: https://shap.readthedocs.io/en/latest/generated/shap.summary_plot.html

  **WHY Each Reference Matters**:
  - `results` and `pipelines` dicts are the EXACT variable names from Task 4 to use for selection
  - SHAP TreeExplainer vs LinearExplainer: must pick correct one based on `hasattr(clf, 'feature_importances_')`
  - `feature_names` is CRITICAL for interpretable SHAP plots (biological names, not column indices)

  **Acceptance Criteria**:
  - [ ] Best model selected by highest mean ROC-AUC
  - [ ] Pipeline fitted on full training data
  - [ ] SHAP explainer is TreeExplainer (XGBoost) or LinearExplainer (LogReg/SVC) -- NOT KernelExplainer
  - [ ] SHAP beeswarm plot rendered with real feature names (not PC1, PC2, etc.)
  - [ ] Top 10 features printed with biological names
  - [ ] Plot saved to `artifacts/shap_beeswarm.png`

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: SHAP beeswarm shows biological feature names
    Tool: Bash
    Preconditions: Cell 5b executed successfully
    Steps:
      1. Check printed output for 'Best model:' line with ROC-AUC value
      2. Check printed output for 'Selected N features:' with list of names
      3. Assert feature names are biological (e.g., pathway names, gene names, clinical vars) NOT 'PC1', 'PC2'
      4. Check 'Top 10 most important features:' list is printed
      5. Verify artifacts/shap_beeswarm.png exists: python -c "import os; assert os.path.exists('artifacts/shap_beeswarm.png')"
    Expected Result: Best model identified, 20 biological feature names listed, top 10 printed, PNG saved
    Failure Indicators: PC-prefixed names, KernelExplainer used, empty feature list, no PNG file
    Evidence: .sisyphus/evidence/task-5-shap-output.txt

  Scenario: SHAP explainer type is correct for model
    Tool: Bash
    Preconditions: Cell 5b code accessible
    Steps:
      1. Parse cell source for explainer creation
      2. Assert TreeExplainer used for XGBoost OR LinearExplainer for LogReg/SVC
      3. Assert KernelExplainer is NOT used anywhere
    Expected Result: Correct explainer type for selected model
    Failure Indicators: KernelExplainer imported or instantiated
    Evidence: .sisyphus/evidence/task-5-explainer-type.txt
  ```

  **Commit**: NO (groups with Task 9)

- [ ] 6. ROC Curve, Confusion Matrix, and Model Comparison Bar Chart

  **What to do**:
  Append the following NEW CELLS to the notebook (after cells from Task 5):

  **Cell 6a (markdown)**: `### 6.5 Visualization: Model Performance`
  Explain: visualizing the best model's ROC curve (from CV predictions), confusion matrix, and a side-by-side comparison of all 3 models.

  **Cell 6b (code)**: ROC Curve using cross_val_predict:
  ```python
  from sklearn.model_selection import cross_val_predict
  from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
  import matplotlib.pyplot as plt
  
  # Get CV probability predictions for ROC curve
  y_proba = cross_val_predict(best_pipeline, X_combined, y_train, cv=cv, method='predict_proba')[:, 1]
  y_pred = cross_val_predict(best_pipeline, X_combined, y_train, cv=cv)
  
  # ROC Curve
  fpr, tpr, _ = roc_curve(y_train, y_proba)
  roc_auc_val = auc(fpr, tpr)
  
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))
  
  # Plot 1: ROC Curve
  axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'{best_name} (AUC = {roc_auc_val:.3f})')
  axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
  axes[0].set_xlabel('False Positive Rate')
  axes[0].set_ylabel('True Positive Rate')
  axes[0].set_title(f'ROC Curve ({best_name})')
  axes[0].legend(loc='lower right')
  
  # Plot 2: Confusion Matrix
  cm = confusion_matrix(y_train, y_pred)
  disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Resp', 'Responder'])
  disp.plot(ax=axes[1], cmap='Blues', colorbar=False)
  axes[1].set_title(f'Confusion Matrix ({best_name})')
  
  # Plot 3: Model Comparison Bar Chart
  model_names = list(results.keys())
  roc_vals = [float(results[n]['ROC-AUC'].split(' ')[0]) for n in model_names]
  f1_vals = [float(results[n]['F1'].split(' ')[0]) for n in model_names]
  ba_vals = [float(results[n]['Bal. Acc.'].split(' ')[0]) for n in model_names]
  
  x = np.arange(len(model_names))
  width = 0.25
  axes[2].bar(x - width, roc_vals, width, label='ROC-AUC', color='steelblue')
  axes[2].bar(x, f1_vals, width, label='F1', color='coral')
  axes[2].bar(x + width, ba_vals, width, label='Bal. Acc.', color='mediumseagreen')
  axes[2].set_xticks(x)
  axes[2].set_xticklabels(model_names)
  axes[2].set_ylabel('Score')
  axes[2].set_title('Model Comparison')
  axes[2].legend()
  axes[2].set_ylim(0, 1)
  
  plt.tight_layout()
  plt.savefig('artifacts/model_comparison.png', dpi=150, bbox_inches='tight')
  plt.show()
  print(f'ROC-AUC (CV predictions): {roc_auc_val:.3f}')
  ```
  - Uses `cross_val_predict` to get out-of-fold predictions (honest evaluation, no leakage)
  - 3 plots in 1 figure row: ROC curve, confusion matrix, model comparison bar chart
  - SHAP beeswarm is already in Task 5 -- total 4 plots (G5 limit)
  - Saves combined figure to `artifacts/model_comparison.png`
  - Parses results dict from Task 4 for bar chart values

  **Must NOT do**:
  - Do NOT create more than 4 total plots across Tasks 5-6 (G5)
  - Do NOT modify existing cells (G1)
  - Do NOT evaluate on test set (this is CV only)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Multiple matplotlib subplots with careful layout, data extraction from results dict, and cross_val_predict usage require attention to detail.
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - `frontend-ui-ux`: matplotlib, not web UI
    - `playwright`: No browser interaction

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - `01_Model_Development.ipynb` cells around execution_count=58-59 -- existing plot cells (learning curves). Shows matplotlib figure style, figsize, tight_layout pattern used in this notebook.
  - `01_Model_Development.ipynb` cells around execution_count=56 -- existing cross_val_predict usage for confusion matrix.

  **API/Type References**:
  - `best_pipeline` from Task 5 -- fitted Pipeline to use with cross_val_predict
  - `best_name` str from Task 5 -- name for plot titles
  - `results` dict from Task 4 -- contains 'ROC-AUC', 'F1', 'Bal. Acc.' strings to parse for bar chart
  - `X_combined` DataFrame from Task 3, `y_train` Series, `cv` from existing notebook

  **External References**:
  - sklearn ConfusionMatrixDisplay: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
  - sklearn cross_val_predict: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html

  **WHY Each Reference Matters**:
  - Existing plot cells show the exact matplotlib style (figsize, dpi, tight_layout) -- follow for consistency
  - cross_val_predict gives out-of-fold predictions -- honest evaluation without data leakage
  - `results` dict stores metric strings as 'mean ± std' -- need to parse with `.split(' ')[0]` for bar chart float values

  **Acceptance Criteria**:
  - [ ] ROC curve plotted with AUC annotation
  - [ ] Confusion matrix shows 'Non-Resp' and 'Responder' labels
  - [ ] Model comparison bar chart shows 3 models × 3 metrics
  - [ ] Total plots across Tasks 5-6 = exactly 4 (SHAP beeswarm + ROC + CM + bar chart)
  - [ ] Combined figure saved to `artifacts/model_comparison.png`
  - [ ] cross_val_predict used (NOT predict on train -- that would overfit)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: All 3 visualization plots render in single figure
    Tool: Bash
    Preconditions: Cell 6b executed successfully
    Steps:
      1. Check notebook execution output for no matplotlib errors
      2. Verify artifacts/model_comparison.png exists: python -c "import os; assert os.path.exists('artifacts/model_comparison.png')"
      3. Check printed ROC-AUC value is between 0.0 and 1.0
    Expected Result: PNG file saved, ROC-AUC printed, no errors
    Failure Indicators: matplotlib error, missing PNG, NaN ROC-AUC
    Evidence: .sisyphus/evidence/task-6-plots.txt

  Scenario: Total plot count does not exceed 4
    Tool: Bash
    Preconditions: Tasks 5 and 6 cells appended
    Steps:
      1. Count plt.show() calls in new cells (Tasks 5+6)
      2. Assert total <= 4 plot calls
    Expected Result: Exactly 2 plt.show() calls (1 in Task 5 SHAP, 1 in Task 6 combined)
    Failure Indicators: More than 4 distinct plots
    Evidence: .sisyphus/evidence/task-6-plot-count.txt
  ```

  **Commit**: NO (groups with Task 9)

- [ ] 7. Save Improved Pipeline Artifacts

  **What to do**:
  Append the following NEW CELLS to the notebook (after cells from Task 6):

  **Cell 7a (markdown)**: `### 6.6 Save Improved Pipeline`
  Explain: saving the best model pipeline and metadata to a SEPARATE artifact file for deployment.

  **Cell 7b (code)**: Save artifacts:
  ```python
  import joblib
  
  # Set n_jobs=None on all components to avoid RLock serialization error (G12)
  for name, step in best_pipeline.named_steps.items():
      if hasattr(step, 'n_jobs'):
          step.n_jobs = None
  
  # Save improved artifacts (NEVER overwrite existing pipeline_artifacts.joblib)
  improved_artifacts = {
      'pipeline': best_pipeline,
      'model_name': best_name,
      'feature_columns': list(X_combined.columns),  # all ~114 input features
      'selected_features': selected_features,          # k=20 selected by SelectKBest
      'cv_results': results,                           # full CV comparison
      'class_ratio': 106/31,                           # for scale_pos_weight reference
      'n_features_selected': len(selected_features),
      'random_state': RANDOM_STATE,
  }
  
  joblib.dump(improved_artifacts, 'artifacts/pipeline_artifacts_improved.joblib')
  print('Improved artifacts saved to: artifacts/pipeline_artifacts_improved.joblib')
  print(f'Keys: {list(improved_artifacts.keys())}')
  print(f'Model: {best_name}, Features: {len(selected_features)}')
  
  # Verify original artifacts are untouched
  original = joblib.load('artifacts/pipeline_artifacts.joblib')
  print(f'Original artifacts still intact: {list(original.keys())}')
  ```
  - CRITICAL: Set `n_jobs=None` before joblib.dump (G12 -- avoids RLock error)
  - Save to `pipeline_artifacts_improved.joblib` (G2 -- NEVER overwrite original)
  - Include all metadata needed for deployment (feature columns, selected features, model name)
  - Verify original artifacts still loadable after save

  **Must NOT do**:
  - Do NOT overwrite `pipeline_artifacts.joblib` (G2)
  - Do NOT save with `n_jobs=-1` set on any component (G12)
  - Do NOT modify existing cells (G1)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple artifact serialization with known pattern. Only tricky part is n_jobs=None, which is explicitly specified.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Task 6)
  - **Blocks**: Task 8
  - **Blocked By**: Task 6

  **References**:

  **Pattern References**:
  - `01_Model_Development.ipynb` cells around execution_count=59 -- existing artifact saving cell. Shows joblib.dump pattern and dict structure used. NEW artifacts use same structure but with additional keys.
  - `artifacts/pipeline_artifacts.joblib` -- existing saved artifact. Keys include: 'pipeline', 'train_data', 'feature_columns'. Match structure where possible.

  **API/Type References**:
  - `best_pipeline` from Task 5 -- fitted Pipeline to save
  - `best_name` str from Task 5 -- model name for metadata
  - `X_combined` DataFrame from Task 3 -- column list for feature_columns
  - `selected_features` list from Task 5 -- SelectKBest-selected feature names
  - `results` dict from Task 4 -- CV comparison results
  - `RANDOM_STATE` = 42 from existing notebook

  **WHY Each Reference Matters**:
  - Existing artifact cell shows the exact dict-of-objects pattern -- follow for consistency
  - `n_jobs=None` loop is critical: sklearn/xgboost objects with `n_jobs=-1` contain thread pool references that fail during pickle

  **Acceptance Criteria**:
  - [ ] `artifacts/pipeline_artifacts_improved.joblib` created
  - [ ] File is loadable: `joblib.load()` returns dict with expected keys
  - [ ] n_jobs is None on all pipeline components
  - [ ] Original `artifacts/pipeline_artifacts.joblib` still loads correctly
  - [ ] Artifact dict contains: pipeline, model_name, feature_columns, selected_features, cv_results

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Improved artifacts are loadable and complete
    Tool: Bash
    Preconditions: Cell 7b executed successfully
    Steps:
      1. Run: python -c "import joblib; a=joblib.load('artifacts/pipeline_artifacts_improved.joblib'); print(list(a.keys())); assert 'pipeline' in a; assert 'selected_features' in a; print('OK')"
      2. Assert all expected keys present
      3. Assert n_features_selected <= 25
    Expected Result: Dict with keys [pipeline, model_name, feature_columns, selected_features, cv_results, class_ratio, n_features_selected, random_state], all present
    Failure Indicators: FileNotFoundError, KeyError, UnpicklingError (n_jobs RLock)
    Evidence: .sisyphus/evidence/task-7-artifacts-load.txt

  Scenario: Original artifacts not overwritten
    Tool: Bash
    Preconditions: Cell 7b executed
    Steps:
      1. Run: python -c "import joblib; a=joblib.load('artifacts/pipeline_artifacts.joblib'); print(list(a.keys())); print('Original OK')"
      2. Assert original artifact loads without error
      3. Assert original keys match expected (pipeline, train_data, feature_columns, etc.)
    Expected Result: Original artifact loads with its original keys
    Failure Indicators: FileNotFoundError, different keys than expected
    Evidence: .sisyphus/evidence/task-7-original-intact.txt
  ```

  **Commit**: NO (groups with Task 9)

- [ ] 8. Validate Notebook JSON Integrity and Execute

  **What to do**:
  Validate the notebook is structurally valid and all new cells execute without runtime errors.

  **Step 8a**: Validate notebook JSON structure:
  - Run `python -c "import nbformat; nbformat.validate(nbformat.read('01_Model_Development.ipynb', 4)); print('VALID')"`
  - If invalid, fix the JSON structure issue before proceeding

  **Step 8b**: Verify new cells are appended AFTER existing cells:
  - Load both `01_Model_Development.ipynb` and `artifacts/01_Model_Development.ipynb` (backup)
  - Compare first N cells of current vs backup -- must be byte-identical
  - New cells must be appended at the END only

  **Step 8c**: Execute the full notebook:
  - Run `jupyter nbconvert --to notebook --execute 01_Model_Development.ipynb --output 01_Model_Development_executed.ipynb --ExecutePreprocessor.timeout=600`
  - If execution fails, read the error traceback and fix the offending cell
  - Check that all new cells produce expected output (metrics, plots, artifact save confirmation)
  - Delete the executed output notebook after verification

  **Step 8d**: Verify cell count:
  - Compare cell count of current vs backup notebook
  - Current should have more cells than backup (new cells added)
  - Print the difference

  **Must NOT do**:
  - Do NOT modify existing cells to fix issues (G1) -- only fix new cells
  - Do NOT leave `_executed.ipynb` files in the repo

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Full notebook execution with 600s timeout, error diagnosis, and JSON validation requires careful handling and debugging.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Task 7)
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References**:

  **Pattern References**:
  - `artifacts/01_Model_Development.ipynb` -- backup of original notebook for comparison. Cell-by-cell comparison verifies no existing cells were modified.

  **API/Type References**:
  - `nbformat.validate()` -- validates notebook JSON schema
  - `nbformat.read()` -- reads .ipynb as NotebookNode object
  - `jupyter nbconvert --execute` -- runs all cells in order, outputs to new file

  **WHY Each Reference Matters**:
  - Backup notebook is the ground truth for verifying G1 (no existing cell modifications)
  - nbformat validate catches JSON schema errors that would prevent Jupyter from opening the file
  - nbconvert --execute is the definitive test that all cells run without error in sequence

  **Acceptance Criteria**:
  - [ ] `nbformat.validate()` passes
  - [ ] First N cells identical to backup (no existing cells modified)
  - [ ] Full notebook execution completes without error (timeout 600s)
  - [ ] New cells produce expected output (metrics printed, plots rendered)
  - [ ] No stale `_executed.ipynb` files left behind

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Notebook JSON is valid
    Tool: Bash
    Preconditions: All cells appended (Tasks 1-7 complete)
    Steps:
      1. Run: python -c "import nbformat; nbformat.validate(nbformat.read('01_Model_Development.ipynb', 4)); print('VALID')"
      2. Assert output is 'VALID'
    Expected Result: 'VALID' printed, no ValidationError
    Failure Indicators: nbformat.ValidationError, JSONDecodeError
    Evidence: .sisyphus/evidence/task-8-json-valid.txt

  Scenario: Existing cells unmodified
    Tool: Bash
    Preconditions: Backup exists at artifacts/01_Model_Development.ipynb
    Steps:
      1. Run: python -c "
         import nbformat
         orig = nbformat.read('artifacts/01_Model_Development.ipynb', 4)
         curr = nbformat.read('01_Model_Development.ipynb', 4)
         n_orig = len(orig.cells)
         match = all(orig.cells[i].source == curr.cells[i].source for i in range(n_orig))
         print(f'Original cells: {n_orig}, Current cells: {len(curr.cells)}, Match: {match}')
         assert match, 'EXISTING CELLS MODIFIED!'
         print('ALL EXISTING CELLS IDENTICAL')"
      2. Assert 'ALL EXISTING CELLS IDENTICAL' printed
      3. Assert current has more cells than original
    Expected Result: All original cells match, new cells added at end
    Failure Indicators: 'EXISTING CELLS MODIFIED!' assertion error
    Evidence: .sisyphus/evidence/task-8-cells-unmodified.txt

  Scenario: Full notebook executes successfully
    Tool: Bash (timeout 700s)
    Preconditions: Notebook JSON valid, all cells present
    Steps:
      1. Run: jupyter nbconvert --to notebook --execute 01_Model_Development.ipynb --output /tmp/test_executed.ipynb --ExecutePreprocessor.timeout=600
      2. Assert exit code 0
      3. Clean up: rm /tmp/test_executed.ipynb
    Expected Result: Notebook executes fully, exit code 0
    Failure Indicators: Non-zero exit code, CellExecutionError, timeout
    Evidence: .sisyphus/evidence/task-8-execution.txt
  ```

  **Commit**: NO (groups with Task 9)

- [ ] 9. Final Acceptance Criteria Verification and Commit

  **What to do**:
  Run through ALL Definition of Done items and collect evidence. Then commit.

  **Step 9a**: Verify all acceptance criteria:
  - [ ] ROC-AUC >= 0.60 (mean CV) -- parse from notebook execution output
  - [ ] SHAP beeswarm shows real feature names -- check `artifacts/shap_beeswarm.png` exists and feature names are biological
  - [ ] 4 visualization plots render -- verify SHAP + ROC + CM + bar chart all present
  - [ ] `artifacts/pipeline_artifacts_improved.joblib` is loadable -- run joblib.load()
  - [ ] `artifacts/pipeline_artifacts.joblib` still intact -- run joblib.load()
  - [ ] Existing cells unmodified -- confirmed in Task 8
  - [ ] `random_state=42` used in all new cells -- grep new cells

  **Step 9b**: Collect evidence files:
  - Gather all `.sisyphus/evidence/task-*` files from Tasks 1-8
  - Create summary evidence file

  **Step 9c**: Create git commit:
  - Stage: `01_Model_Development.ipynb`, `artifacts/pipeline_artifacts_improved.joblib`
  - Do NOT stage `artifacts/shap_beeswarm.png` or `artifacts/model_comparison.png` (generated outputs)
  - Commit message: `feat(model): improve ML pipeline with genomic integration, multi-classifier CV, and SHAP visualizations`
  - Pre-commit check: `python -c "import nbformat; nbformat.validate(nbformat.read('01_Model_Development.ipynb', 4))"`

  **Must NOT do**:
  - Do NOT commit if any acceptance criterion fails
  - Do NOT stage documentation or plot PNG files
  - Do NOT push to remote

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Final verification requires parsing notebook output, checking multiple file artifacts, and creating a clean git commit. Requires thoroughness.
  - **Skills**: [`git-master`]
    - `git-master`: Clean commit with proper message format, staging only relevant files
  - **Skills Evaluated but Omitted**:
    - `playwright`: No browser verification needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (final task before verification wave)
  - **Blocks**: F1-F4 (Final Verification Wave)
  - **Blocked By**: Task 8

  **References**:

  **Pattern References**:
  - `.sisyphus/evidence/` directory -- all evidence files from Tasks 1-8
  - `artifacts/pipeline_artifacts_improved.joblib` -- improved artifacts from Task 7
  - `artifacts/pipeline_artifacts.joblib` -- original artifacts (verify untouched)

  **API/Type References**:
  - All variables from prior tasks (results, best_name, selected_features)
  - All evidence files from `.sisyphus/evidence/task-*`

  **WHY Each Reference Matters**:
  - Evidence files prove each task completed successfully -- needed for F1 (Plan Compliance Audit)
  - Both artifact files must be verified to confirm G2 (no overwrite)

  **Acceptance Criteria**:
  - [ ] ALL Definition of Done items verified with evidence
  - [ ] ROC-AUC >= 0.60 confirmed in output
  - [ ] Git commit created with correct message and staged files
  - [ ] No extra files staged (no PNGs, no _executed.ipynb)
  - [ ] Evidence files collected in `.sisyphus/evidence/`

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: All acceptance criteria pass
    Tool: Bash
    Preconditions: Tasks 1-8 complete
    Steps:
      1. Run: python -c "import joblib; a=joblib.load('artifacts/pipeline_artifacts_improved.joblib'); print(f'Model: {a["model_name"]}, Features: {a["n_features_selected"]}, Keys: {list(a.keys())}')"
      2. Assert model_name is one of ['L1 LogReg', 'Linear SVC', 'XGBoost']
      3. Assert n_features_selected <= 25
      4. Verify ROC-AUC >= 0.60 from cv_results in artifact
    Expected Result: Valid model with <=25 features and ROC-AUC >= 0.60
    Failure Indicators: Wrong model name, >25 features, ROC-AUC < 0.60
    Evidence: .sisyphus/evidence/task-9-final-check.txt

  Scenario: Git commit is clean
    Tool: Bash
    Preconditions: Commit created
    Steps:
      1. Run: git log -1 --format='%s'
      2. Assert message starts with 'feat(model):'
      3. Run: git diff --name-only HEAD~1
      4. Assert only 01_Model_Development.ipynb and artifacts/pipeline_artifacts_improved.joblib are changed
    Expected Result: Clean commit with exactly 2 files
    Failure Indicators: Extra files, wrong message format, PNG files committed
    Evidence: .sisyphus/evidence/task-9-git-commit.txt
  ```

  **Commit**: YES
  - Message: `feat(model): improve ML pipeline with genomic integration, multi-classifier CV, and SHAP visualizations`
  - Files: `01_Model_Development.ipynb`, `artifacts/pipeline_artifacts_improved.joblib`
  - Pre-commit: `python -c "import nbformat; nbformat.validate(nbformat.read('01_Model_Development.ipynb', 4))"`

---

## Final Verification Wave (MANDATORY -- after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection -> fix -> re-run.

- [ ] F1. **Plan Compliance Audit** -- `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read notebook cells, check output). For each "Must NOT Have": search notebook for forbidden patterns -- reject with cell number if found. Check evidence files exist in `.sisyphus/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** -- `unspecified-high`
  Run `python -c "import nbformat; nbformat.validate(nbformat.read('01_Model_Development.ipynb', 4))"`. Review all new cells for: bare `except:`, `print()` without context, hardcoded paths, missing `random_state`. Check for data leakage: `fit_transform` on test data, feature selection outside CV. Check for AI slop: excessive comments, over-abstraction.
  Output: `Notebook valid [PASS/FAIL] | Leakage [CLEAN/N issues] | Code quality [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** -- `unspecified-high`
  Execute the entire notebook via `jupyter nbconvert --to notebook --execute 01_Model_Development.ipynb --output executed.ipynb --ExecutePreprocessor.timeout=600`. Parse output cells for: ROC-AUC values printed, SHAP plot execution success, artifact save success. Check that existing cells still produce same outputs. Save evidence.
  Output: `Execution [PASS/FAIL] | ROC-AUC [value] | Plots [N/N rendered] | Artifacts [PASS/FAIL] | VERDICT`

- [ ] F4. **Scope Fidelity Check** -- `deep`
  Compare first N cells of notebook with backup in `artifacts/01_Model_Development.ipynb` -- must be identical. Verify only new cells were added at end. Check new cells do not exceed scope (no deployment changes, no new files beyond artifact). Count features used in final model -- must be <=25.
  Output: `Existing cells [IDENTICAL/MODIFIED] | Scope [CLEAN/N violations] | Features [N <= 25] | VERDICT`

---

## Commit Strategy

- **Single commit after all tasks complete**: `feat(model): improve ML pipeline with genomic integration, nested CV, and SHAP visualizations`
  - Files: `01_Model_Development.ipynb`, `artifacts/pipeline_artifacts_improved.joblib`
  - Pre-commit: `python -c "import nbformat; nbformat.validate(nbformat.read('01_Model_Development.ipynb', 4))"`

---

## Success Criteria

### Verification Commands
```bash
# Notebook is valid JSON
python -c "import nbformat; nbformat.validate(nbformat.read('01_Model_Development.ipynb', 4)); print('VALID')"

# Improved artifacts are loadable
python -c "import joblib; a=joblib.load('artifacts/pipeline_artifacts_improved.joblib'); print(list(a.keys()))"

# Original artifacts still intact
python -c "import joblib; a=joblib.load('artifacts/pipeline_artifacts.joblib'); print('Original OK:', list(a.keys()))"

# Notebook executes without error (full run)
jupyter nbconvert --to notebook --execute 01_Model_Development.ipynb --ExecutePreprocessor.timeout=600
```

### Final Checklist
- [ ] ROC-AUC >= 0.60 (mean CV) printed in notebook output
- [ ] All "Must Have" items present
- [ ] All "Must NOT Have" items absent
- [ ] 4 visualization plots render without error
- [ ] SHAP shows biological feature names (not PC numbers)
- [ ] Existing cells unmodified
- [ ] Improved artifacts saved and loadable
- [ ] `random_state=42` used consistently
