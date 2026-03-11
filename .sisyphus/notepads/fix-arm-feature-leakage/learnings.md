# Learnings - fix-arm-feature-leakage

Session: ses_321c1b67cffeq2jfz10l76N0IE
Started: 2026-03-11T23:00:00Z

## Conventions & Patterns

_Accumulated wisdom from task execution will be appended here_

---

## [2026-03-12] Task 1: MICE Response Leakage Fix

### Summary
Fixed target variable leakage in MICE imputation pipeline. Response was being included in the imputation kernel, which is a critical ML violation.

### Changes Made
- **Cell 26 (index 25)**: Removed `train_clinic[['Response']]` from pd.concat in `train_clinical_encoded`
  - Before: 10 columns (9 features + Response)
  - After: 9 columns (9 features only)
  
- **Cell 28 (index 27)**: Added Response re-attachment after MICE.complete_data()
  - MICE now runs on predictors only (arm_df, binary_encoded, ordinal_encoded, age_scaled)
  - Response is concatenated back to maintain downstream compatibility (Cell 58)

### Verification Steps Passed
1. Cell 25: Response removed from encoding pipeline ✓
2. Cell 27: Response re-attached after imputation ✓
3. No downstream breaks (Cell 58 still has Response for dropping) ✓

### Key Learning
**MICE (Multiple Imputation by Chained Equations) should NEVER include the target variable.** The MICE algorithm uses all included columns to impute missing values. If the target (Response) is in the kernel:
- MICE learns the relationship between missing predictors and the outcome
- This means the imputation process uses "future information" (the answer we're trying to predict)
- This constitutes target leakage—the model indirectly learns from the outcome during preprocessing

### Pattern: Imputation ≠ Feature Engineering
- Imputation is a DATA PREPROCESSING step (fit on training, apply to test)
- Always exclude targets from imputation kernels
- Re-attach targets afterward if downstream code expects them

### Architecture Note
The pipeline structure is now:
1. Encode features (without target)
2. Impute missing values in features
3. Attach target
4. Continue with feature selection, scaling, and modeling

This ensures the MICE kernel only sees the feature space.


## Task 2: Arm Removal & Ancillary Fixes (2026-03-12)

### Changes Made
1. **Cell 56**: Genomic path `train_nivo` → `train` (224 patients vs 137)
2. **Cell 58**: Added Arm column removal (drops Arm_EVEROLIMUS, Arm_NIVOLUMAB)
3. **Cell 60**: scale_pos_weight hardcoded 3.42 → dynamic computation
4. **Cell 66**: class_ratio hardcoded 106/31 → dynamic computation
5. **Markdown updates** (Cells 53, 55, 57, 59, 65): Feature counts corrected

### Feature Impact
- Before: ~116 features (incl. 2 Arm cols, Response target)
- After: ~113 features (excl. Arm, Response already dropped)
- Clinical: 9 → 7 (removed Arm_EVEROLIMUS, Arm_NIVOLUMAB)

### Key Learning: Why Arm Leakage Matters
**Arm = treatment assignment (Nivolumab vs Everolimus)**
- NOT a biological characteristic of the patient
- Including it teaches: "This drug works better" (treatment effect)
- We need: "This biology predicts response" (biological signal)
- Arm becomes top SHAP feature (blocks interpretability)
- Model fails to generalize to new treatments

### Data Row Mismatch Fix
- Genomic was loading from train_nivo (137 rows)
- All other data: 224 rows
- Result: 87 patients with zero-imputed genomics
- **Fix**: Load from train/ (all 224 rows)

### Class Ratio Corrections
- Was: scale_pos_weight=3.42, class_ratio=106/31
- Based on: Stale training split
- Now: Computed dynamically from y_train
- Actual ratio: ~5.4 (not 3.42)

### Verification
- All 4 code fixes verified with assertions
- All 5 markdown fixes verified
- 4 evidence files created in .sisyphus/evidence/
