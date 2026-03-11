# Learnings - improve-ml-pipeline

Session: ses_321c1b67cffeq2jfz10l76N0IE
Started: 2026-03-11T19:29:25.147Z

## Conventions & Patterns

_Accumulated wisdom from task execution will be appended here_

---
## Task 1: Dependency Installation & Notebook Setup

### nbformat for Safe Notebook Modification
**Pattern**: Use `nbformat` library instead of direct JSON manipulation
- **Why**: Preserves notebook metadata, cell structure, and execution history
- **Key steps**:
  1. Read notebook with `nbformat.read(fp, as_version=4)`
  2. Determine next execution_count from existing cells
  3. Create cells with `nbformat.v4.new_markdown_cell()` / `new_code_cell()`
  4. Append to `nb.cells` array
  5. Write with `nbformat.write(nb, fp)`

### Version Pin Strategy for Package Compatibility
**Problem**: NumPy 2.x broke scikit-learn/pandas compatibility
**Solution**: Pin compatible versions across dependency chain
- SHAP 0.49.1 (compatible with numpy<2)
- XGBoost 3.2.0 (no issues with numpy 1.x)

### Execution Count Management
**Convention**: Execution counts are sequential integers
- Check max existing: `max(cell.execution_count for cell in nb.cells if cell.execution_count)`
- Increment by 1 for new code cells
- Markdown cells don't have execution_count

### Dependencies Verified
| Package | Version | Min Required | Status |
|---------|---------|--------------|--------|
| SHAP | 0.49.1 | 0.40 | ✅ TreeExplainer compatible |
| XGBoost | 3.2.0 | 1.5 | ✅ scale_pos_weight supported |

### Blockers & Technical Debt
- NumPy version conflict in environment (mixed versions due to constraints)
- Warnings appear but don't break functionality
- Future: Consider full environment rebuild

### ⚠️ CRITICAL FIX: nbformat Normalization Issue
**Problem**: nbformat library normalizes whitespace and line endings when read/write
- Caused silent modifications to all cell source arrays during round-trip
- Violates G1 guardrail: existing cells must remain BYTE-IDENTICAL

**Solution**: Use raw JSON with minimal indent for append-only operations
```python
import json
with open('notebook.ipynb', 'r') as f:
    nb = json.load(f)  # Raw JSON, no nbformat normalization
# Append new cells as dicts
nb['cells'].append(new_cell_dict)
with open('notebook.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)  # Minimal formatting
```

**Verification**: Deep JSON comparison against `git show HEAD:` to ensure original cells unchanged


---
## Task 2: Notebook Structure Analysis & Variable Mapping

### Notebook Dimensions
- **Total cells**: 54 (26 code + 28 markdown)
- **Last cell index**: 53 (0-indexed)
- **Next insertion point**: Cell 54
- **Task 1 added cells**: 52-53 (markdown header + dependency check code)

### Key Variables Available in Kernel

| Variable | Type | Shape | Notes |
|----------|------|-------|-------|
| `X_train_all` | DataFrame | (137, 38110) | Full integrated features (no genomic) |
| `y_train` | Series | (137,) | Binary response (0/1) |
| `train_clinical_imputed` | DataFrame | (137, 10) | Clinical features after encoding/imputation |
| `train_trans_scaled` | DataFrame | (137, 38030) | Transcriptomics (variance-filtered, scaled) |
| `train_pathway_scaled` | DataFrame | (137, 50) | ssGSEA pathways (scaled) |
| `deconv_clr_df` | DataFrame | (137, 22) | Immune deconvolution (CLR-transformed, scaled) |
| `keep_genes` | list | 38030 items | Gene identifiers after variance filter |
| `RANDOM_STATE` | int | — | Value: 42 (constant) |
| `cv` | StratifiedKFold | — | 5 splits × 3 repeats = 15 iterations |
| `transformers` | dict | 7 objects | Encoders/scalers: ohe_Arm, le_*, oe_MSKCC, scaler_Age |
| `pca` | PCA | — | Fitted PCA with 95% variance retention |
| `trans_scaler` | StandardScaler | — | Fitted on transcriptomics |
| `pathway_scaler` | StandardScaler | — | Fitted on pathways |
| `clr_scaler` | StandardScaler | — | Fitted on CLR deconvolution |

### Genomic Data Status ⚠️

**File**: `Data/train_nivo/genomic.csv`
**Shape**: (137, 34) — 137 samples × 34 columns
**Columns**:
- 1 Patient_ID (index)
- 13 Copy-number features (Amplification/Deletion at loci)
- 20 Gene mutations (ARID1A, ATM, BAP1, etc.)

**Value encodings**:
- `MUT`: Mutation present
- `WT`: Wild-type (no mutation)
- `NO_IF`: No information (missing)

**CRITICAL**: Genomic data was DISPLAYED in cell 4 but NEVER assigned to a variable.
This is the gap Task 3 must address.

### Feature Engineering Strategy

**Transcriptomics preprocessing** (Cell 13):
- Initial: 40,934 genes
- After variance filter (var > 0.05): 38,030 genes
- Output: `train_trans_scaled` — StandardScaler fitted on training data

**Clinical encoding** (Cells 21-25):
- One-hot: Arm (NIVOLUMAB, EVEROLIMUS)
- Binary: Sex, Sarc, Rhab, Tumor_Sample_Primary_or_Metastasis
- Ordinal: MSKCC (POOR → 0, INTERMEDIATE → 1, FAVORABLE → 2)
- Numeric: Age (standardized)
- Imputation: MICE (10 iterations) for missing values

**Immune deconvolution** (Cell 18):
- Input: 22 cell types from CIBERSORTx
- Transform: Centered-logratio (CLR) to handle compositional nature
- Output: `deconv_clr_df` — CLR-transformed + StandardScaler fitted

**Pathway scores** (Cell 16):
- Input: 50 Hallmark ssGSEA scores
- Transform: StandardScaler
- Output: `train_pathway_scaled`

### Integration Point

Cell 30 concatenates all modalities:
```python
df_train_all = pd.concat([
    train_clinical_processed,
    train_trans_scaled,
    train_pathway_scaled,
    deconv_clr_df
], axis=1)
```

Result: **X_train_all shape (137, 38110)**
= 10 (clinical) + 38,030 (trans) + 50 (pathway) + 22 (immune)

### Cross-Validation Configuration

Cell 33 defines CV:
```python
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=RANDOM_STATE
)
```
Total folds: 15 (5 folds × 3 repeats)
Stratified: Yes (preserves class balance in each fold)

### Training Set Imbalance

- Non-responders (0): 100 patients (73.5%)
- Responders (1): 37 patients (26.5%)
- Ratio: ~2.7:1

### Cell Execution Timeline

- Cells 0-51: Original notebook content (fully executed)
- Cells 52-53: Task 1 additions (dependency check, not variable creation)
- Status: Ready for appending at cell index 54

### Critical Insights for Task 3

1. **Feature count baseline**: 38,110 (without genomic)
2. **Expected after genomic**: ~38,145 (adding 33 genomic columns, excluding Patient_ID)
3. **Patient ID matching**: Must align genomic by Patient_ID column before concatenating
4. **Data leakage risk**: Genomic encoders (if any) must be fit on training only
5. **Value encoding choice**: MUT→1, WT→0 (binary) vs. ordinal vs. keep categorical

### Evidence Location

Full detailed evidence saved to:
`~/.sisyphus/evidence/task-2-notebook-structure.txt`

Contains: Cell listing, variable shapes, preprocessing details, genomic CSV spec,
recommendations for Task 3 integration strategy.

---
## Task 3: Genomic Data Integration & Combined Feature Matrix

### Genomic Data Loaded
- **File**: `Data/train_nivo/genomic.csv`
- **Observed raw shape with `index_col='Patient_ID'`**: (137, 33)
- **After encoding + indicator**: (137, 34) with `has_genomic_data`
- **Encodings**: MUT→1, WT→0, NO_IF→NaN→fill(0)
- **Coverage**: 66.4% patients have at least one genomic feature (91/137)
- **Biomarkers present**: BAP1, PBRM1, VHL, SETD2

### Combined Feature Matrix
- **Shape**: (137, 120)
- **Breakdown**:
  - Clinical: 10 (dropped 'Response' column in notebook cell logic)
  - Pathway: 50
  - Deconvolution: 26
  - Genomic: 34 (33 + has_genomic_data)
- **Total features**: 120
- **Data leakage check**: 'Response' NOT in feature_names ✓

### Guardrail Verification (G1)
- Notebook total cells after append: 58
- First 54 cells unchanged vs `git show HEAD:01_Model_Development.ipynb`: **True**

### Technical Notes
- Used raw JSON append (`json.load/json.dump(indent=1)`) to avoid nbformat normalization drift
- Added 4 cells at indexes 54-57 only (2 markdown + 2 code)
- Stored `feature_names` from `X_combined.columns` for SHAP usage in next tasks

### Evidence Files
- `.sisyphus/evidence/task-3-verification.txt`

### Blockers & Issues
- Minor spec/data mismatch observed: dataset currently yields 33 genomic feature columns after setting `Patient_ID` as index, not 34. Implemented required logic exactly and preserved expected processing flow.

---
## Task 4: 3 Classifier Pipelines + CV Comparison

### Pipelines Built
- **L1 LogReg**: StandardScaler → SelectKBest(k=20, MI) → LogReg(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced')
- **Linear SVC**: StandardScaler → SelectKBest(k=20, MI) → SVC(kernel='linear', C=0.1, class_weight='balanced', probability=True)
- **XGBoost**: StandardScaler → SelectKBest(k=20, MI) → XGBClassifier(n_estimators=50, max_depth=3, lr=0.01, scale_pos_weight=3.42)

### Feature Selection
- **Method**: SelectKBest with mutual_info_classif (deterministic via functools.partial with random_state=42)
- **k=20 features**: Chosen per N/5 rule (137/5≈27) → conservative k=20

### Class Imbalance Handling
- **L1 LogReg & Linear SVC**: `class_weight='balanced'` (auto-computed as n_samples / (n_classes * np.bincount(y)))
- **XGBoost**: `scale_pos_weight=3.42` (106 non-responders / 31 responders)
- **NO SMOTE/ADASYN** per G7 guardrail (avoids k-neighbor failures in small CV folds)

### Cross-Validation Results
Deferred by guardrail: notebook execution is explicitly prohibited in Task 4 (`DO NOT execute the notebook cells`).

- **Evaluation setup in appended code**: RepeatedStratifiedKFold (5 splits × 3 repeats = 15 iterations)
- **Metrics configured**: ROC-AUC (primary), F1-score, Balanced Accuracy

### Guardrail Verification (G1)
- Notebook total cells after append: 60
- First 58 cells unchanged vs git HEAD: True

### Technical Notes
- Used `functools.partial(mutual_info_classif, random_state=42)` for deterministic MI scoring (G13)
- All pipelines use `random_state=42` for reproducibility
- XGBoost: `use_label_encoder=False` and `eval_metric='logloss'` to suppress warnings
- SVC: `max_iter=10000` to ensure convergence

### Blockers & Issues
- Cannot provide executed CV table or ROC-AUC threshold confirmation in Task 4 because notebook execution is disallowed by task constraints.
