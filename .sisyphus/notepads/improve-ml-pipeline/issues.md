# Issues - improve-ml-pipeline

Session: ses_321c1b67cffeq2jfz10l76N0IE
Started: 2026-03-11T19:29:25.147Z

## Problems & Gotchas

_Issues encountered and their solutions will be appended here_

---

## 2026-03-11: F3 Manual QA - Critical Failure in Cell 60

### Issue
Notebook execution fails at Cell 60 (3-classifier cross-validation) with:
```
ValueError: Input X contains NaN. SelectKBest does not accept missing values
```

### Root Cause
Cell 58 creates `X_combined` by concatenating 4 dataframes:
1. `train_clinical_imputed` 
2. `train_pathway_scaled`
3. `deconv_clr_df`
4. `genomic_filled`

While `genomic_filled` is properly filled with `.fillna(0)` in Cell 56, at least ONE of the other source dataframes still contains NaN values. The code assumes all source data is clean but does not verify or handle residual NaN values.

### Impact
- Cell 60: FAILS before CV completes (all 15 folds fail)
- Cell 62 (SHAP): NOT EXECUTED (blocked)
- Cell 64 (Visualizations): NOT EXECUTED (blocked)
- Cell 66 (Artifact save): NOT EXECUTED (blocked)
- NO artifacts created
- NO ROC-AUC metrics available
- **CANNOT complete QA** until this is fixed

### Required Fix
Add NaN handling in Cell 58 or new Cell 59:

**Option 1: Simple imputation (recommended)**
```python
# After X_combined creation in Cell 58
from sklearn.impute import SimpleImputer

# Check for NaN
nan_count = X_combined.isna().sum().sum()
if nan_count > 0:
    print(f'⚠️  Found {nan_count} NaN values in X_combined')
    print(f'Columns with NaN: {X_combined.columns[X_combined.isna().any()].tolist()}')
    
    # Impute with 0 (safe for binary, scaled, and CLR features)
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_combined_clean = pd.DataFrame(
        imputer.fit_transform(X_combined),
        columns=X_combined.columns,
        index=X_combined.index
    )
    X_combined = X_combined_clean
    print('✓ NaN values imputed with 0')
```

**Option 2: Fail-fast with informative error**
```python
nan_count = X_combined.isna().sum().sum()
if nan_count > 0:
    nan_cols = X_combined.columns[X_combined.isna().any()].tolist()
    raise ValueError(
        f'X_combined contains {nan_count} NaN values in columns: {nan_cols}. '
        'Please verify that all source dataframes are properly imputed.'
    )
```

### Pre-existing Bugs Found in Original Notebook
While debugging, discovered these bugs in cells 1-52:

1. **Cell 36** (Learning curve): `NameError: name 'pipeline' is not defined`
   - Original notebook references undefined variable
   - Blocks execution of section 6.3
   
2. **Cell 48** (ADASYN pipeline): Missing imports
   - Uses `ADASYN`, `SelectKBest`, `mutual_info_classif`, `LogisticRegression`, `cross_validate`
   - But only imports `Pipeline` from imblearn
   - These should be imported at cell start:
     ```python
     from imblearn.over_sampling import ADASYN
     from sklearn.feature_selection import SelectKBest, mutual_info_classif
     from sklearn.linear_model import LogisticRegression
     from sklearn.model_selection import cross_validate
     ```

### Status
**BLOCKED** - Cannot proceed with QA until Cell 58/59 adds NaN handling.

### Partial Success
Cells 54, 56, 58 executed successfully:
- ✓ Dependencies verified (SHAP 0.51.0, XGBoost 3.2.0)
- ✓ Genomic data loaded (137×34, 66.4% coverage)
- ✓ X_combined created (224×115 features)

But data quality validation was MISSING, causing downstream failure.

## F3 Manual QA - Regression Analysis (2026-03-11 22:10)

### Discovered Issues

**1. Cell 31 Execution Failure**
- Error: `ValueError: cross_val_predict only works for partitions`
- Root cause: `RepeatedStratifiedKFold` creates overlapping indices (repeated folds), incompatible with `cross_val_predict`
- Impact: ROC curve, confusion matrix, and bar chart plots not generated
- Workaround: `model_comparison.png` exists from earlier execution (timestamp 21:16)
- Fix: Replace `cv=cv` (RepeatedStratifiedKFold) with `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

**2. Cells 19, 23-25 Cascade Failure**
- Cells: 19 (learning curve), 23 (pipeline def), 24 (feature importance), 25 (artifact save)
- Error: `NameError: name 'ADASYN' is not defined` (Cell 23) → cascade `NameError: name 'pipeline' is not defined` (19, 24, 25)
- Root cause: Cell 23 uses `ADASYN(random_state=RANDOM_STATE)` without importing it
- Timeline:
  - Commits 074f9bd → a788e5a: No errors in cells 1-26
  - Current HEAD (7473a9d): Errors present in cells 19, 23-25
- Hypothesis: Notebook was re-executed ("Run All") exposing missing import that was previously satisfied by deleted/modified cell
- Fix: Add `from imblearn.over_sampling import ADASYN` to Cell 23 imports

### Impact Assessment

**Not Affected:**
- Original artifacts (`pipeline_artifacts.joblib`) - unchanged (git status clean, last modified 21:52)
- New model cells (27-30, 32) - executed successfully
- Core functionality: ROC-AUC 0.661 achieved, SHAP plot generated, improved artifacts saved

**Affected:**
- Cells 19, 23-25 cannot execute until ADASYN import added
- Cell 31 cannot execute until CV object changed from RepeatedStratifiedKFold

### Lessons Learned

1. **Jupyter notebook hygiene**: Always verify imports in cells where classes are used, even if "it worked before"
2. **Cross-validation compatibility**: `cross_val_predict` requires non-overlapping partitions → incompatible with `RepeatedKFold` variants
3. **Artifact timestamp forensics**: File timestamps reveal which outputs are stale vs fresh
4. **Git archaeology**: Use `git show <commit>:<file>` to trace when regressions were introduced

